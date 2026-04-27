from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
from tqdm import tqdm

from csbdeep.models import BaseConfig
from csbdeep.internals.blocks import unet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage, IS_KERAS_3_PLUS, BACKEND as K
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from scipy.ndimage import zoom
from packaging.version import Version
from collections.abc import Iterable
from skimage.draw import polygon
from skimage.transform import resize
from itertools import product

keras = keras_import()
Input, Conv2D, MaxPooling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D')
Model = keras_import('models', 'Model')

from .base import StarDistBase, StarDistDataBase, _tf_version_at_least
from ..sample_patches import sample_patches
from ..utils import edt_prob, _normalize_grid, mask_to_categorical
from ..geometry import star_dist, dist_to_coord, polygons_to_label
from ..nms import non_maximum_suppression, non_maximum_suppression_sparse

_gen_rtype = list if IS_TF_1 else tuple

class StarDistData2D(StarDistDataBase):

    def __init__(self, X, Y, batch_size, n_rays, length,
                 n_classes=None, classes=None,
                 patch_size=(256,256), b=32, grid=(1,1), shape_completion=False, augmenter=None, foreground_prob=0, **kwargs):

        super().__init__(X=X, Y=Y, n_rays=n_rays, grid=grid,
                         n_classes=n_classes, classes=classes,
                         batch_size=batch_size, patch_size=patch_size, length=length,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            if not all(b % g == 0 for g in self.grid):
                raise ValueError(f"'shape_completion' requires that crop size {b} ('train_completion_crop' in config) is evenly divisible by all grid values {self.grid}")
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'


    def __getitem__(self, i):
        idx = self.batch(i)
        arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                 patch_size=self.patch_size, n_samples=1,
                                 valid_inds=self.get_valid_inds(k)) for k in idx]

        if self.n_channel is None:
            X, Y = list(zip(*[(x[0][self.b],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1)[self.b], y[0]) for y,*x in arrays]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X,Y))))

        mask_neg_labels = tuple(y[self.b][self.ss_grid[1:3]] < 0 for y in Y)
        has_neg_labels = any(m.any() for m in mask_neg_labels)
        if has_neg_labels:
            mask_neg_labels = np.stack(mask_neg_labels)
            # set negative label pixels to 0 (background)
            Y = tuple(np.maximum(y, 0) for y in Y)

        prob = np.stack([edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in Y])
        # prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])
        # prob = prob[self.ss_grid]

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            _dist     = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist      = _dist[self.ss_grid]
            dist_mask = np.stack([edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in Y_cleared])
        else:
            # directly subsample with grid
            dist      = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode, grid=self.grid) for lbl in Y])
            dist_mask = prob

        X = np.stack(X)
        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        # subsample wth given grid
        # dist_mask = dist_mask[self.ss_grid]
        # prob      = prob[self.ss_grid]

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(dist.shape[:-1]+(self.n_rays+1,), np.float32)
        dist_and_mask[...,:-1] = dist
        dist_and_mask[...,-1:] = dist_mask

        if has_neg_labels:
            prob[mask_neg_labels] = -1  # set to -1 to disable loss

        # note: must return tuples in keras 3 (cf. https://stackoverflow.com/a/78158487)
        if self.n_classes is None:
            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask))
        else:
            prob_class = np.stack(tuple((mask_to_categorical(y[self.b], self.n_classes, self.classes[k]) for y,k in zip(Y, idx))))

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(prob_class, (1,)+tuple(1/g for g in self.grid)+(1,), order=0)

            if has_neg_labels:
                prob_class[mask_neg_labels] = -1  # set to -1 to disable loss

            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask, prob_class))



class Config2D(BaseConfig):
    """Configuration for a :class:`StarDist2D` model.

    Parameters
    ----------
    axes : str or None
        Axes of the input images.
    n_rays : int
        Number of radial directions for the star-convex polygon.
        Recommended to use a power of 2 (default: 32).
    n_channel_in : int
        Number of channels of given input image (default: 1).
    grid : (int,int)
        Subsampling factors (must be powers of 2) for each of the axes.
        Model will predict on a subsampled grid for increased efficiency and larger field of view.
    n_classes : None or int
        Number of object classes to use for multi-class prediction (use None to disable)
    backbone : str
        Name of the neural network architecture to be used as backbone.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_pool : (int,int)
        Maxpooling size for all (U-Net) convolution layers.
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    unet_* : *
        Additional parameters for U-net backbone.
    train_shape_completion : bool
        Train model to predict complete shapes for partially visible objects at image boundary.
    train_completion_crop : int
        If 'train_shape_completion' is set to True, specify number of pixels to crop at boundary of training patches.
        Should be chosen based on (largest) object sizes.
    train_patch_size : (int,int)
        Size of patches to be cropped from provided training images.
    train_background_reg : float
        Regularizer to encourage distance predictions on background regions to be 0.
    train_foreground_only : float
        Fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels.
    train_sample_cache : bool
        Activate caching of valid patch regions for all training images (disable to save memory for large datasets)
    train_dist_loss : str
        Training loss for star-convex polygon distances ('mse' or 'mae').
    train_loss_weights : tuple of float
        Weights for losses relating to (probability, distance)
    train_epochs : int
        Number of training epochs.
    train_steps_per_epoch : int
        Number of parameter update steps per epoch.
    train_learning_rate : float
        Learning rate for training.
    train_batch_size : int
        Batch size for training.
    train_n_val_patches : int
        Number of patches to be extracted from validation images (``None`` = one patch per image).
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress.
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
    use_gpu : bool
        Indicate that the data generator should use OpenCL to do computations on the GPU.

        .. _ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    """

    def __init__(self, axes='YX', n_rays=32, n_channel_in=1, grid=(1,1), n_classes=None, backbone='unet', **kwargs):
        """See class docstring."""

        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1+n_rays)

        # directly set by parameters
        self.n_rays                    = int(n_rays)
        self.grid                      = _normalize_grid(grid,2)
        self.backbone                  = str(backbone).lower()
        self.n_classes                 = None if n_classes is None else int(n_classes)

        # default config (can be overwritten by kwargs below)
        if self.backbone == 'unet':
            self.unet_n_depth          = 3
            self.unet_kernel_size      = 3,3
            self.unet_n_filter_base    = 32
            self.unet_n_conv_per_depth = 2
            self.unet_pool             = 2,2
            self.unet_activation       = 'relu'
            self.unet_last_activation  = 'relu'
            self.unet_batch_norm       = False
            self.unet_dropout          = 0.0
            self.unet_expansion        = 2
            self.unet_prefix           = ''
            self.net_conv_after_unet   = 128
        else:
            # TODO: resnet backbone for 2D model?
            raise ValueError("backbone '%s' not supported." % self.backbone)

        # net_mask_shape not needed but kept for legacy reasons
        if backend_channels_last():
            self.net_input_shape       = None,None,self.n_channel_in
            self.net_mask_shape        = None,None,1
        else:
            self.net_input_shape       = self.n_channel_in,None,None
            self.net_mask_shape        = 1,None,None

        self.train_shape_completion    = False
        self.train_completion_crop     = 32
        self.train_patch_size          = 256,256
        self.train_background_reg      = 1e-4
        self.train_foreground_only     = 0.9
        self.train_sample_cache        = True

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = (1,0.2) if self.n_classes is None else (1,0.2,1)
        self.train_class_weights       = (1,1) if self.n_classes is None else (1,)*(self.n_classes+1)
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 100
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 4
        self.train_n_val_patches       = None
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        # keras.__version__ was removed in tensorflow 2.13.0
        min_delta_key = 'epsilon' if Version(getattr(keras, '__version__', '9.9.9'))<=Version('2.1.5') else 'min_delta'
        self.train_reduce_lr           = {'factor': 0.5, 'patience': 40, min_delta_key: 0}

        self.use_gpu                   = False

        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        # FIXME: put into is_valid()
        if not len(self.train_loss_weights) == (2 if self.n_classes is None else 3):
            raise ValueError(f"train_loss_weights {self.train_loss_weights} not compatible with n_classes ({self.n_classes}): must be 3 weights if n_classes is not None, otherwise 2")

        if not len(self.train_class_weights) == (2 if self.n_classes is None else self.n_classes+1):
            raise ValueError(f"train_class_weights {self.train_class_weights} not compatible with n_classes ({self.n_classes}): must be 'n_classes + 1' weights if n_classes is not None, otherwise 2")



class StarDist2D(StarDistBase):
    """StarDist2D model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=Config2D(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        output_prob = Conv2D(                 1, (1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv2D(self.config.n_rays, (1,1), name='dist', padding='same', activation='linear')(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base

            output_prob_class  = Conv2D(self.config.n_classes+1, (1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    def train(self, X, Y, validation_data, classes='auto', augmenter=None, seed=None, epochs=None, steps_per_epoch=None, workers=1):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Input images
        Y : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Label masks
            Positive pixel values denote object instance ids (0 for background).
            Negative values can be used to turn off all losses for the corresponding pixels (e.g. for regions that haven't been labeled).
        classes (optional): 'auto' or iterable of same length as X
             label id -> class id mapping for each label mask of Y if multiclass prediction is activated (n_classes > 0)
             list of dicts with label id -> class id (1,...,n_classes)
             'auto' -> all objects will be assigned to the first non-background class,
                       or will be ignored if config.n_classes is None
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`) or triple (if multiclass)
            Tuple (triple if multiclass) of X,Y,[classes] validation data.
        augmenter : None or callable
            Function with expected signature ``xt, yt = augmenter(x, y)``
            that takes in a single pair of input/label image (x,y) and returns
            the transformed images (xt, yt) for the purpose of data augmentation
            during training. Not applied to validation images.
            Example:
            def simple_augmenter(x,y):
                x = x + 0.05*np.random.normal(0,1,x.shape)
                return x,y
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        classes = self._parse_classes_arg(classes, len(X))

        if not self._is_multiclass() and classes is not None:
            warnings.warn("Ignoring given classes as n_classes is set to None")

        isinstance(validation_data,(list,tuple)) or _raise(ValueError())
        if self._is_multiclass() and len(validation_data) == 2:
            validation_data = tuple(validation_data) + ('auto',)
        ((len(validation_data) == (3 if self._is_multiclass() else 2))
            or _raise(ValueError(f'len(validation_data) = {len(validation_data)}, but should be {3 if self._is_multiclass() else 2}')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        div_by = self._axes_div_by(axes)
        [(p-2*b) % d == 0 or _raise(ValueError(
            "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if self.config.train_shape_completion else
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            n_rays           = self.config.n_rays,
            patch_size       = self.config.train_patch_size,
            grid             = self.config.grid,
            shape_completion = self.config.train_shape_completion,
            b                = self.config.train_completion_crop,
            use_gpu          = self.config.use_gpu,
            foreground_prob  = self.config.train_foreground_only,
            n_classes        = self.config.n_classes,
            sample_ind_cache = self.config.train_sample_cache,
        )
        worker_kwargs = dict(workers=workers, use_multiprocessing=workers>1)
        if IS_KERAS_3_PLUS:
            data_kwargs['keras_kwargs'] = worker_kwargs
            fit_kwargs = {}
        else:
            fit_kwargs = worker_kwargs

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = self._parse_classes_arg(validation_data[2], n_data_val) if self._is_multiclass() else None
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        _data_val = StarDistData2D(validation_data[0],validation_data[1], classes=classes_val, batch_size=n_take, length=1, **data_kwargs)
        data_val = _data_val[0]

        # expose data generator as member for general diagnostics
        self.data_train = StarDistData2D(X, Y, classes=classes, batch_size=self.config.train_batch_size,
                                         augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)

        if self.config.train_tensorboard:
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            channel = axes_dict(self.config.axes)['C']
            output_slices = [[slice(None)]*4,[slice(None)]*4]
            output_slices[1][1+channel] = slice(0,(self.config.n_rays//_n)*_n, self.config.n_rays//_n)
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)]*4]
                output_slices[2][1+channel] = slice(1,1+(self.config.n_classes//_n)*_n, self.config.n_classes//_n)

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*4,None]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks):
                self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=data_val, log_dir=str(self.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, output_slices=output_slices))

        fit = self.keras_model.fit_generator if (IS_TF_1 and not IS_KERAS_3_PLUS) else self.keras_model.fit
        history = fit(iter(self.data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      **fit_kwargs,
                      callbacks=self.callbacks, verbose=1,
                      # set validation batchsize to training batchsize (only works for tf >= 2.2)
                      **(dict(validation_batch_size = self.config.train_batch_size) if _tf_version_at_least("2.2.0") else {}))
        self._training_finished()

        return history

    @staticmethod
    def refine(labels, polys, thr=0.5, w_winner=2, progress=False):
        """shape refinement"""
        thr = float(thr)
        assert 0 <= thr <= 1, f"required: 0 <= {thr} <= 1"
        if thr == 1:
            # to include only pixels where all polys agree
            # because we take mask > thr below
            thr -= np.finfo(float).eps
        nms = polys["nms"]
        obj_ind = np.flatnonzero(nms["suppressed"] == -1)
        assert np.allclose(nms["scores"][obj_ind], sorted(nms["scores"][obj_ind])[::-1])
        mask = np.zeros_like(labels)
        # mask_soft = np.zeros_like(labels, float)

        # TODO: use prob/scores for weighting?
        # TODO: use mask that weights pixels on distance to poly boundary?
        for k, i in tqdm(
                zip(range(len(obj_ind), 0, -1), reversed(obj_ind)),
                total=len(obj_ind),
                disable=(not progress),
        ):
            polys_i = nms["coord"][i: i + 1]  # winner poly after nms
            polys_i_suppressed = nms["coord"][nms["suppressed"] == i]  # suppressed polys by winner
            # array of all polys (first winner, then all suppressed)
            polys_i = np.concatenate([polys_i, polys_i_suppressed], axis=0)
            # bounding slice around all polys wrt image
            ss = tuple(
                slice(max(int(np.floor(start)), 0), min(int(np.ceil(stop)), w))
                for start, stop, w in zip(
                    np.min(polys_i, axis=(0, 2)), np.max(polys_i, axis=(0, 2)), labels.shape
                )
            )
            # shape of image crop/region that contains all polys
            shape_i = tuple(s.stop - s.start for s in ss)
            # offset of image region
            offset = np.array([s.start for s in ss]).reshape(2, 1)
            # voting weights for polys
            n_i = len(polys_i)
            # vote weight of winning poly (1 = same vote as each suppressed poly)
            weight_winner = w_winner
            # define and normalize weights for all polys
            polys_i_weights = np.ones(n_i)
            polys_i_weights[0] = weight_winner
            # polys_i_weights = np.array([weight_winner if j==0 else max(0,n_i-weight_winner)/(n_i-1) for j in range(n_i)])
            polys_i_weights = polys_i_weights / np.sum(polys_i_weights)
            # display(polys_i_weights)
            assert np.allclose(np.sum(polys_i_weights), 1)
            # merge by summing weighted poly masks
            mask_i = np.zeros(shape_i, float)
            for p, w in zip(polys_i, polys_i_weights):
                ind = polygon(*(p - offset), shape=shape_i)
                mask_i[ind] += w
            # write refined shape for instance i back to new label image
            # refined shape are all pixels with accumulated votes >= threshold
            mask[ss][mask_i > thr] = k
            # mask_soft[ss][mask_i>0] += mask_i[mask_i>0]

        return mask  # , mask_soft

    @staticmethod
    def rot90(x, k=1, roll=True):
        """Rotate stardist cnn predictions by multiples of 90 degrees."""
        from stardist import ray_angles

        k = (k + 4) % 4
        # print(k)
        assert x.ndim in (2, 3)
        if x.ndim == 2 or roll == False:
            # rotate 2D image or 2D+channel
            return np.rot90(x, k)
        # dist image has radial distances as 3rd dimension
        # -> need to roll values
        deg_roll = (-90 * k) % 360
        rad_roll = np.deg2rad(deg_roll)
        # print(deg_roll)
        n_rays = x.shape[2]
        rays = ray_angles(n_rays)
        n_roll = [i for i, v in enumerate(rays) if np.isclose(v, rad_roll)]
        assert len(n_roll) == 1, (rays, rad_roll)
        n_roll = n_roll[0]
        z = np.rot90(x, k, axes=(0, 1))  # rotate spatial axes
        z = np.roll(z, n_roll, axis=2)  # roll polar axis
        return z

    @staticmethod
    def flip(x, doit=True, reverse=True):
        """Flip stardist cnn predictions."""
        assert x.ndim in (2, 3)
        if not doit:
            return x
        if x.ndim == 2 or reverse == False:
            return np.flipud(x)
        # dist image has radial distances as 3rd dimension
        # -> need to reverse values
        z = np.flipud(x)
        z = np.concatenate((z[..., 0:1], z[..., :0:-1]), axis=-1)
        return z

    @staticmethod
    def crop_center(x, crop_shape):
        """Crop an array at the centre with specified dimensions."""
        orig_shape = x.shape
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
        return x

    @staticmethod
    def count_classes(y, classes=range(1, 7), crop=(224, 224)):
        assert y.ndim == 3 and y.shape[-1] == 2
        if crop is not None:
            y = StarDist2D.crop_center(y, crop)
        return tuple(len(np.unique(y[..., 0] * (y[..., 1] == i))) - 1 for i in classes)

    def predict_nms(
            self,
            x,
            normalize=False,
            aggregate_prob_class=True,
            no_background_class=True,
            crop_counts=True,
            test_time_augment=False,
            refine_shapes=False,
            tta_merge=dict(prob=np.median, dist=np.mean, prob_class=np.mean),
            return_details=False,
            kwargs_instances=None,
            **kwargs,
    ):
        """ main prediction function to be used from several parts of the code

        Accepts a list of models for ensemble prediction

        model: StardDist2D model (or list of models)
        aggregate_prob_class: take the mean of prob_class per instance (instead of single point)
        no_background_class:  ignore background class in prob_class
        crop_counts:          crop labels to 224x224 before counting
        test_time_augment:    True or maximal number of tta per model (=8 if True)
        refine_shapes:        False or dict of refine kwargs (empty dict for default values)
        Returns
        -------
        u, counts

        u:        class map arrays of shape (256,256,2)
        counts:   cell counts,  tuple of length 6

        """
        from skimage.morphology import remove_small_objects

        if kwargs_instances is None:
            kwargs_instances = {}

        model = self.keras_model

        # model should be a list of models
        if not isinstance(model, Iterable):
            model = (model,)

        # if more than one model is given, ensure all have same outputs
        if not all([m.keras_model.output_shape == model[0].keras_model.output_shape for m in model]):
            raise ValueError(
                "Cannot combine model output shapes (models have different number of rays?)"
            )

        # use average thresholds of the models
        kwargs_instances.setdefault("nms_thresh", np.mean([m.thresholds.nms for m in model]))
        kwargs_instances.setdefault("prob_thresh", np.mean([m.thresholds.prob for m in model]))

        warnings.warn(f"{kwargs_instances}")

        if test_time_augment == False:
            # print('no tta')
            # no augmentation
            augs = ((0, False),)
            model_augs = tuple(product(model, augs))
        elif test_time_augment in (True, -1):
            # print('full tta')
            # 8 augmentations (4 rotations x 2 flips)
            augs = tuple(product((0, 1, 2, 3), (False, True)))
            model_augs = tuple(product(model, augs))
        else:
            # print(f'partial tta {test_time_augment}')
            augs = tuple(product((0, 1, 2, 3), (False, True)))

            # augs = augs[:test_time_augment]
            # model_augs  = tuple(product(model, augs ))

            # picking random flip/roations
            # rng = np.random.RandomState(42)
            # aug_idx = tuple(rng.choice(np.arange(len(augs)), min(test_time_augment, len(augs)), replace=False) for _ in model)

            aug_idx = (([1, 5, 0]), ([3, 7, 0]), ([0, 6, 3]), ([5, 2, 7]))
            model_augs = tuple((m, augs[i]) for m, idx in zip(model, aug_idx) for i in idx)

        warnings.warn(f"combining {len(model)} models -> total {len(model_augs)} predictions")

        def _preprocess(x):
            if normalize:
                x = x.astype(np.float32) / 255
            return x

        prob, dist, prob_class = zip(
            *[
                m.predict(self.flip(self.rot90(_preprocess(x), k, False), f, False), **kwargs)
                for m, (k, f) in model_augs
            ]
        )

        # undo augmentations for predictions
        prob = [self.rot90(self.flip(v, f), -k) for (m, (k, f)), v in zip(model_augs, prob)]
        dist = [self.rot90(self.flip(v, f), -k) for (m, (k, f)), v in zip(model_augs, dist)]
        prob_class = [
            self.rot90(self.flip(v, f, False), -k, False) for (m, (k, f)), v in zip(model_augs, prob_class)
        ]

        # merge predictions
        prob = tta_merge["prob"](np.stack(prob), axis=0)
        dist = tta_merge["dist"](np.stack(dist), axis=0)
        prob_class = tta_merge["prob_class"](np.stack(prob_class), axis=0)
        prob_class /= np.sum(prob_class, axis=-1, keepdims=True)

        u, res = model[0]._instances_from_prediction(
            x.shape[:2], prob, dist, prob_class=prob_class, **kwargs_instances
        )

        if refine_shapes is not False:
            assert "nms" in res
            u = self.refine(u, res, **refine_shapes)

        u_cls = np.zeros(u.shape, np.uint16)

        n_objects = len(res["prob"])

        cls = dict(zip(range(1, n_objects + 1), res["class_id"]))

        # u = remove_small_objects(u,10)

        if any(g > 1 for g in model[0].config.grid):
            prob = resize(prob, u.shape, order=1)
            prob_class = resize(prob_class, u.shape + prob_class.shape[-1:], order=1)

        if aggregate_prob_class:
            # take the sum of class probabilities
            pc_weighted = np.expand_dims(prob, -1) * prob_class
            if no_background_class:
                pc_weighted[..., 0] = -1

            for r in regionprops(u):
                m = u[r.slice] == r.label
                class_id = np.argmax(np.sum(pc_weighted[r.slice][m], 0))
                u_cls[r.slice][m] = class_id
                cls[r.label] = class_id
        else:
            # only take center class prob
            for r in regionprops(u):
                m = u[r.slice] == r.label
                u_cls[r.slice][m] = cls[r.label]

        out = np.stack([u, u_cls], axis=-1)

        class_count = self.count_classes(out, classes=range(1, 7), crop=(224, 224) if crop_counts else None)

        if return_details:
            return out, class_count, cls, prob_class
        else:
            return out, class_count



    # def _instances_from_prediction_old(self, img_shape, prob, dist,points = None, prob_class = None,  prob_thresh=None, nms_thresh=None, overlap_label = None, **nms_kwargs):
    #     from stardist.geometry.geom2d import _polygons_to_label_old, _dist_to_coord_old
    #     from stardist.nms import _non_maximum_suppression_old

    #     if prob_thresh is None: prob_thresh = self.thresholds.prob
    #     if nms_thresh  is None: nms_thresh  = self.thresholds.nms
    #     if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

    #     coord = _dist_to_coord_old(dist, grid=self.config.grid)
    #     inds = _non_maximum_suppression_old(coord, prob, grid=self.config.grid,
    #                                    prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
    #     labels = _polygons_to_label_old(coord, prob, inds, shape=img_shape)
    #     # sort 'inds' such that ids in 'labels' map to entries in polygon dictionary entries
    #     inds = inds[np.argsort(prob[inds[:,0],inds[:,1]])]
    #     # adjust for grid
    #     points = inds*np.array(self.config.grid)

    #     res_dict = dict(coord=coord[inds[:,0],inds[:,1]], points=points, prob=prob[inds[:,0],inds[:,1]])

    #     if prob_class is not None:
    #         prob_class = np.asarray(prob_class)
    #         res_dict.update(dict(class_prob = prob_class))

    #     return labels, res_dict


    def _instances_from_prediction(self, img_shape, prob, dist, points=None, prob_class=None, prob_thresh=None, nms_thresh=None, overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi  class prediction
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms
        if overlap_label is not None: raise NotImplementedError("overlap_label not supported for 2D yet!")

        # sparse prediction
        if points is not None:
            points, probi, disti, indsi = non_maximum_suppression_sparse(dist, prob, points, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                prob_class = prob_class[indsi]
            nms = None

        # dense prediction
        else:
            points, probi, disti, nms = non_maximum_suppression(dist, prob, grid=self.config.grid,
                                                                prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5):
            #   1. re-scale points (origins of polygons)
            #   2. re-scale coordinates (computed from distances) of (zero-origin) polygons
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X' and 'Y'")
            rescale = (1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,2)
        else:
            rescale = (1,1)

        if return_labels:
            labels = polygons_to_label(disti, points, prob=probi, shape=img_shape, scale_dist=rescale)
        else:
            labels = None

        coord = dist_to_coord(disti, points, scale_dist=rescale)

        if nms is not None:
            nms['coord'] = dist_to_coord(nms['dist'], nms['points'], scale_dist=rescale)

        res_dict = dict(coord=coord, points=points, prob=probi, nms=nms)

        # multi class prediction
        if prob_class is not None:
            prob_class = np.asarray(prob_class)
            # ignore background for class_ids
            class_id = 1+np.argmax(prob_class[...,1:], axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict


    def _axes_div_by(self, query_axes):
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.unet_pool) == len(self.config.grid)
        div_by = dict(zip(
            self.config.axes.replace('C',''),
            tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
        ))
        return tuple(div_by.get(a,1) for a in query_axes)


    # def _axes_tile_overlap(self, query_axes):
    #     self.config.backbone == 'unet' or _raise(NotImplementedError())
    #     query_axes = axes_check_and_normalize(query_axes)
    #     assert len(self.config.unet_pool) == len(self.config.grid) == len(self.config.unet_kernel_size)
    #     # TODO: compute this properly when any value of grid > 1
    #     # all(g==1 for g in self.config.grid) or warnings.warn('FIXME')
    #     overlap = dict(zip(
    #         self.config.axes.replace('C',''),
    #         tuple(tile_overlap(self.config.unet_n_depth + int(np.log2(g)), k, p)
    #               for p,k,g in zip(self.config.unet_pool,self.config.unet_kernel_size,self.config.grid))
    #     ))
    #     return tuple(overlap.get(a,0) for a in query_axes)


    @property
    def _config_class(self):
        return Config2D
