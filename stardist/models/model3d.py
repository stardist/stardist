from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import math
from tqdm import tqdm


from csbdeep.models import BaseConfig
from csbdeep.internals.blocks import conv_block3, unet_block, resnet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage
from distutils.version import LooseVersion
from scipy.ndimage import zoom
from skimage.measure  import regionprops
keras = keras_import()
K = keras_import('backend')
Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Concatenate = keras_import('layers', 'Input', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Add', 'Concatenate')
Model = keras_import('models', 'Model')

from .base import StarDistBase, StarDistDataBase, _tf_version_at_least
from ..sample_patches import sample_patches
from ..utils import edt_prob, _normalize_grid, mask_to_categorical
from ..matching import relabel_sequential
from ..geometry import star_dist3D, polyhedron_to_label
from ..rays3d import Rays_GoldenSpiral, rays_from_json
from ..nms import non_maximum_suppression_3d, non_maximum_suppression_3d_sparse


class StarDistData3D(StarDistDataBase):

    def __init__(self, X, Y, batch_size, rays, length,
                 n_classes=None, classes=None,
                 patch_size=(128,128,128), grid=(1,1,1), anisotropy=None, augmenter=None, foreground_prob=0, **kwargs):
        # TODO: support shape completion as in 2D?

        super().__init__(X=X, Y=Y, n_rays=len(rays), grid=grid,
                         classes=classes, n_classes=n_classes,
                         batch_size=batch_size, patch_size=patch_size, length=length,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.rays = rays
        self.anisotropy = anisotropy
        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'
        # re-use arrays
        if self.batch_size > 1:
            self.out_X = np.empty((self.batch_size,)+tuple(self.patch_size)+(() if self.n_channel is None else (self.n_channel,)), dtype=np.float32)
            patch_size_grid = tuple((p-1)//g+1 for p,g in zip(self.patch_size,self.grid))
            self.out_edt_prob = np.empty((self.batch_size,)+patch_size_grid, dtype=np.float32)
            self.out_star_dist3D = np.empty((self.batch_size,)+patch_size_grid+(len(self.rays),), dtype=np.float32)
            if self.n_classes is not None:
                self.out_prob_class = np.empty((self.batch_size,)+tuple(self.patch_size)+(self.n_classes+1,), dtype=np.float32)


    def __getitem__(self, i):
        idx = self.batch(i)
        arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                 patch_size=self.patch_size, n_samples=1,
                                 valid_inds=self.get_valid_inds(k)) for k in idx]

        if self.n_channel is None:
            X, Y = list(zip(*[(x[0],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1), y[0]) for y,*x in arrays]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X,Y))))

        if len(Y) == 1:
            X = X[0][np.newaxis]
        else:
            X = np.stack(X, out=self.out_X[:len(Y)])
        if X.ndim == 4: # input image has no channel axis
            X = np.expand_dims(X,-1)

        tmp = [edt_prob(lbl, anisotropy=self.anisotropy)[self.ss_grid[1:]] for lbl in Y]
        if len(Y) == 1:
            prob = tmp[0][np.newaxis]
        else:
            prob = np.stack(tmp, out=self.out_edt_prob[:len(Y)])

        tmp = [star_dist3D(lbl, self.rays, mode=self.sd_mode, grid=self.grid) for lbl in Y]
        if len(Y) == 1:
            dist = tmp[0][np.newaxis]
        else:
            dist = np.stack(tmp, out=self.out_star_dist3D[:len(Y)])

        prob = dist_mask = np.expand_dims(prob, -1)

        # append dist_mask to dist as additional channel
        dist = np.concatenate([dist,dist_mask],axis=-1)

        if self.n_classes is None:
            return [X], [prob,dist]
        else:
            tmp = [mask_to_categorical(y, self.n_classes, self.classes[k]) for y,k in zip(Y, idx)]
            # TODO: downsample here before stacking?
            if len(Y) == 1:
                prob_class = tmp[0][np.newaxis]
            else:
                prob_class = np.stack(tmp, out=self.out_prob_class[:len(Y)])

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(prob_class, (1,)+tuple(1/g for g in self.grid)+(1,), order=0)

            return [X], [prob,dist, prob_class]



class Config3D(BaseConfig):
    """Configuration for a :class:`StarDist3D` model.

    Parameters
    ----------
    axes : str or None
        Axes of the input images.
    rays : Rays_Base, int, or None
        Ray factory (e.g. Ray_GoldenSpiral).
        If an integer then Ray_GoldenSpiral(rays) will be used
    n_channel_in : int
        Number of channels of given input image (default: 1).
    grid : (int,int,int)
        Subsampling factors (must be powers of 2) for each of the axes.
        Model will predict on a subsampled grid for increased efficiency and larger field of view.
    n_classes : None or int
        Number of object classes to use for multi-class predection (use None to disable)
    anisotropy : (float,float,float)
        Anisotropy of objects along each of the axes.
        Use ``None`` to disable only for (nearly) isotropic objects shapes.
        Also see ``utils.calculate_extents``.
    backbone : str
        Name of the neural network architecture to be used as backbone.
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int,int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_pool : (int,int,int)
        Maxpooling size for all (U-Net) convolution layers.
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    unet_* : *
        Additional parameters for U-net backbone.
    resnet_n_blocks : int
        Number of ResNet blocks.
    resnet_kernel_size : (int,int,int)
        Convolution kernel size for all ResNet blocks.
    resnet_n_filter_base : int
        Number of convolution kernels (feature channels) for ResNet blocks.
        (Number is doubled after every downsampling, see ``grid``.)
    net_conv_after_resnet : int
        Number of filters of the extra convolution layer after ResNet (0 to disable).
    resnet_* : *
        Additional parameters for ResNet backbone.
    train_patch_size : (int,int,int)
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
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress.
    train_n_val_patches : int
        Number of patches to be extracted from validation images (``None`` = one patch per image).
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.
    use_gpu : bool
        Indicate that the data generator should use OpenCL to do computations on the GPU.

        .. _ReduceLROnPlateau: https://keras.io/api/callbacks/reduce_lr_on_plateau/
    """

    def __init__(self, axes='ZYX', rays=None, n_channel_in=1, grid=(1,1,1), n_classes=None, anisotropy=None, backbone='unet', **kwargs):

        if rays is None:
            if 'rays_json' in kwargs:
                rays = rays_from_json(kwargs['rays_json'])
            elif 'n_rays' in kwargs:
                rays = Rays_GoldenSpiral(kwargs['n_rays'])
            else:
                rays = Rays_GoldenSpiral(96)
        elif np.isscalar(rays):
            rays = Rays_GoldenSpiral(rays)

        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1+len(rays))

        # directly set by parameters
        self.n_rays                    = len(rays)
        self.grid                      = _normalize_grid(grid,3)
        self.anisotropy                = anisotropy if anisotropy is None else tuple(anisotropy)
        self.backbone                  = str(backbone).lower()
        self.rays_json                 = rays.to_json()
        self.n_classes                 = None if n_classes is None else int(n_classes)

        if 'anisotropy' in self.rays_json['kwargs']:
            if self.rays_json['kwargs']['anisotropy'] is None and self.anisotropy is not None:
                self.rays_json['kwargs']['anisotropy'] = self.anisotropy
                print("Changing 'anisotropy' of rays to %s" % str(anisotropy))
            elif self.rays_json['kwargs']['anisotropy'] != self.anisotropy:
                warnings.warn("Mismatch of 'anisotropy' of rays and 'anisotropy'.")

        # default config (can be overwritten by kwargs below)
        if self.backbone == 'unet':
            self.unet_n_depth            = 2
            self.unet_kernel_size        = 3,3,3
            self.unet_n_filter_base      = 32
            self.unet_n_conv_per_depth   = 2
            self.unet_pool               = 2,2,2
            self.unet_activation         = 'relu'
            self.unet_last_activation    = 'relu'
            self.unet_batch_norm         = False
            self.unet_dropout            = 0.0
            self.unet_prefix             = ''
            self.net_conv_after_unet     = 128
        elif self.backbone == 'resnet':
            self.resnet_n_blocks         = 4
            self.resnet_kernel_size      = 3,3,3
            self.resnet_kernel_init      = 'he_normal'
            self.resnet_n_filter_base    = 32
            self.resnet_n_conv_per_block = 3
            self.resnet_activation       = 'relu'
            self.resnet_batch_norm       = False
            self.net_conv_after_resnet   = 128
        else:
            raise ValueError("backbone '%s' not supported." % self.backbone)

        # net_mask_shape not needed but kept for legacy reasons
        if backend_channels_last():
            self.net_input_shape       = None,None,None,self.n_channel_in
            self.net_mask_shape        = None,None,None,1
        else:
            self.net_input_shape       = self.n_channel_in,None,None,None
            self.net_mask_shape        = 1,None,None,None

        # self.train_shape_completion    = False
        # self.train_completion_crop     = 32
        self.train_patch_size          = 128,128,128
        self.train_background_reg      = 1e-4
        self.train_foreground_only     = 0.9
        self.train_sample_cache        = True

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = (1,0.2) if self.n_classes is None else (1,0.2,1)
        self.train_class_weights       = (1,1) if self.n_classes is None else (1,)*(self.n_classes+1)
        self.train_epochs              = 400
        self.train_steps_per_epoch     = 100
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 1
        self.train_n_val_patches       = None
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        self.train_reduce_lr           = {'factor': 0.5, 'patience': 40, min_delta_key: 0}

        self.use_gpu                   = False

        # remove derived attributes that shouldn't be overwritten
        for k in ('n_dim', 'n_channel_out', 'n_rays', 'rays_json'):
            try: del kwargs[k]
            except KeyError: pass

        self.update_parameters(False, **kwargs)

        # FIXME: put into is_valid()
        if not len(self.train_loss_weights) == (2 if self.n_classes is None else 3):
            raise ValueError(f"train_loss_weights {self.train_loss_weights} not compatible with n_classes ({self.n_classes}): must be 3 weights if n_classes is not None, otherwise 2")

        if not len(self.train_class_weights) == (2 if self.n_classes is None else self.n_classes+1):
            raise ValueError(f"train_class_weights {self.train_class_weights} not compatible with n_classes ({self.n_classes}): must be 'n_classes + 1' weights if n_classes is not None, otherwise 2")


class StarDist3D(StarDistBase):
    """StarDist3D model.

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

    def __init__(self, config=Config3D(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        if self.config.backbone == "unet":
            return self._build_unet()
        elif self.config.backbone == "resnet":
            return self._build_resnet()
        else:
            raise NotImplementedError(self.config.backbone)


    def _build_unet(self):
        assert self.config.backbone == 'unet'
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv3D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling3D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    def _build_resnet(self):
        assert self.config.backbone == 'resnet'
        n_filter = self.config.resnet_n_filter_base
        resnet_kwargs = dict (
            kernel_size        = self.config.resnet_kernel_size,
            n_conv_per_block   = self.config.resnet_n_conv_per_block,
            batch_norm         = self.config.resnet_batch_norm,
            kernel_initializer = self.config.resnet_kernel_init,
            activation         = self.config.resnet_activation,
        )

        input_img = Input(self.config.net_input_shape, name='input')

        layer = input_img
        layer = Conv3D(n_filter, (7,7,7), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)
        layer = Conv3D(n_filter, (3,3,3), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)

        pooled = np.array([1,1,1])
        for n in range(self.config.resnet_n_blocks):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            if any(p > 1 for p in pool):
                n_filter *= 2
            layer = resnet_block(n_filter, pool=tuple(pool), **resnet_kwargs)(layer)

        layer_base = layer

        if self.config.net_conv_after_resnet > 0:
            layer = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                           name='features', padding='same', activation=self.config.resnet_activation)(layer_base)

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(layer)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(layer)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_resnet > 0:
                layer_class  = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                                      name='features_class', padding='same', activation=self.config.resnet_activation)(layer_base)
            else:
                layer_class  = layer_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(layer_class)
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
        div_by = self._axes_div_by(axes)
        [p % d == 0 or _raise(ValueError(
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            rays             = rays_from_json(self.config.rays_json),
            grid             = self.config.grid,
            patch_size       = self.config.train_patch_size,
            anisotropy       = self.config.anisotropy,
            use_gpu          = self.config.use_gpu,
            foreground_prob  = self.config.train_foreground_only,
            n_classes        = self.config.n_classes,
            sample_ind_cache = self.config.train_sample_cache,
        )

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = self._parse_classes_arg(validation_data[2], n_data_val) if self._is_multiclass() else None
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        _data_val = StarDistData3D(validation_data[0],validation_data[1], classes=classes_val, batch_size=n_take, length=1, **data_kwargs)
        data_val = _data_val[0]

        # expose data generator as member for general diagnostics
        self.data_train = StarDistData3D(X, Y, classes=classes, batch_size=self.config.train_batch_size,
                                         augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)

        if self.config.train_tensorboard:
            # only show middle slice of 3D inputs/outputs
            input_slices, output_slices = [[slice(None)]*5], [[slice(None)]*5,[slice(None)]*5]
            i = axes_dict(self.config.axes)['Z']
            channel = axes_dict(self.config.axes)['C']
            _n_in  = _data_val.patch_size[i] // 2
            _n_out = _data_val.patch_size[i] // (2 * (self.config.grid[i] if self.config.grid is not None else 1))
            input_slices[0][1+i] = _n_in
            output_slices[0][1+i] = _n_out
            output_slices[1][1+i] = _n_out
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            output_slices[1][1+channel] = slice(0,(self.config.n_rays//_n)*_n, self.config.n_rays//_n)
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)]*5]
                output_slices[2][1+channel] = slice(1,1+(self.config.n_classes//_n)*_n, self.config.n_classes//_n)

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.input_slices = input_slices
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*5,None]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks):
                self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=data_val, log_dir=str(self.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, input_slices=input_slices, output_slices=output_slices))

        fit = self.keras_model.fit_generator if IS_TF_1 else self.keras_model.fit
        history = fit(iter(self.data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      workers=workers, use_multiprocessing=workers>1,
                      callbacks=self.callbacks, verbose=1,
                      # set validation batchsize to training batchsize (only works in tf 2.x)
                      **(dict(validation_batch_size = self.config.train_batch_size) if _tf_version_at_least("2.2.0") else {}))
        self._training_finished()

        return history


    def _instances_from_prediction(self, img_shape, prob, dist, points=None, prob_class=None, prob_thresh=None, nms_thresh=None, overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi  class prediction
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)

        # sparse prediction
        if points is not None:
            points, probi, disti, indsi = non_maximum_suppression_3d_sparse(dist, prob, points, rays, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                prob_class = prob_class[indsi]

        # dense prediction
        else:
            points, probi, disti = non_maximum_suppression_3d(dist, prob, rays, grid=self.config.grid,
                                                              prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5,Z=1.0):
            #   1. re-scale points (origins of polyhedra)
            #   2. re-scale vectors of rays object (computed from distances)
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale and 'Z' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X', 'Y', and 'Z'")
            rescale = (1/scale['Z'],1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,3)
            rays = rays.copy(scale=rescale)
        else:
            rescale = (1,1,1)

        if return_labels:
            labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label, verbose=verbose)

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        else:
            labels = None

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            # build the list of class ids per label via majority vote
            # zoom prob_class to img_shape
            # prob_class_up = zoom(prob_class,
            #                      tuple(s2/s1 for s1, s2 in zip(prob_class.shape[:3], img_shape))+(1,),
            #                      order=0)
            # class_id, label_ids = [], []
            # for reg in regionprops(labels):
            #     m = labels[reg.slice]==reg.label
            #     cls_id = np.argmax(np.mean(prob_class_up[reg.slice][m], axis = 0))
            #     class_id.append(cls_id)
            #     label_ids.append(reg.label)
            # # just a sanity check whether labels where in sorted order
            # assert all(x <= y for x,y in zip(label_ids, label_ids[1:]))
            # res_dict.update(dict(classes = class_id))
            # res_dict.update(dict(labels = label_ids))
            # self.p = prob_class_up

            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict


    def _axes_div_by(self, query_axes):
        if self.config.backbone == "unet":
            query_axes = axes_check_and_normalize(query_axes)
            assert len(self.config.unet_pool) == len(self.config.grid)
            div_by = dict(zip(
                self.config.axes.replace('C',''),
                tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
            ))
            return tuple(div_by.get(a,1) for a in query_axes)
        elif self.config.backbone == "resnet":
            grid_dict = dict(zip(self.config.axes.replace('C',''), self.config.grid))
            return tuple(grid_dict.get(a,1) for a in query_axes)
        else:
            raise NotImplementedError()


    @property
    def _config_class(self):
        return Config3D
