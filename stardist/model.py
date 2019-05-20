from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
import warnings
import math
from tqdm import tqdm

from distutils.version import LooseVersion
import keras
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import Adam

from csbdeep.models import BaseConfig, BaseModel
from csbdeep.internals.blocks import unet_block
from csbdeep.internals.predict import tile_iterator, tile_overlap
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.data import Resizer

from .utils import star_dist, edt_prob, _normalize_grid, dist_to_coord, polygons_to_label
from .nms import non_maximum_suppression
from skimage.segmentation import clear_border



if not backend_channels_last():
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored" % K.image_data_format()
    )



def masked_loss(mask, penalty):
    def _loss(d_true, d_pred):
        return K.mean(mask * penalty(d_pred - d_true), axis=-1)
    return _loss

def masked_loss_mae(mask):
    return masked_loss(mask, K.abs)

def masked_loss_mse(mask):
    return masked_loss(mask, K.square)



class StarDistPadAndCropResizer(Resizer):
    # TODO: check correctness
    def __init__(self, grid, mode='reflect', **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.kwargs = kwargs

    def before(self, x, axes, axes_div_by):
        assert all(a%g==0 for g,a in zip((self.grid.get(a,1) for a in axes), axes_div_by))
        axes = axes_check_and_normalize(axes,x.ndim)
        def _split(v):
            return 0, v # only pad at the end
        self.pad = {
            a : _split((div_n-s%div_n)%div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs)
        self.padded_shape = dict(zip(axes,x_pad.shape))
        if 'C' in self.padded_shape: del self.padded_shape['C']
        return x_pad

    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes,x.ndim)
        assert all(s_pad == s * g for s,s_pad,g in zip(x.shape,
                                                       (self.padded_shape.get(a,_s) for a,_s in zip(axes,x.shape)),
                                                       (self.grid.get(a,1) for a in axes)))
        # print(self.padded_shape)
        # print(self.pad)
        # print(self.grid)
        crop = tuple (
            slice(0, -(math.floor(p[1]/g)) if p[1]>=g else None)
            for p,g in zip((self.pad.get(a,(0,0)) for a in axes),(self.grid.get(a,1) for a in axes))
        )
        # print(crop)
        return x[crop]



class StarDistData(Sequence):

    def __init__(self, X, Y, batch_size, n_rays, patch_size=(256,256), b=32, grid=(1,1), shape_completion=False, same_patches=False):
        """
        Parameters
        ----------
        same_patches : bool
            Set to true for validation data to always get the same patch for each image
        """

        # TODO: simple augmentations (rotation & flips)
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.n_rays = n_rays
        self.patch_size = patch_size
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid)
        self.perm = np.random.permutation(len(self.X))
        self.shape_completion = bool(shape_completion)
        self.same_patches = bool(same_patches)

        if self.shape_completion and b > 0:
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def _random_patches(self, shapes, idx):
        def _single_patch(shape,i):
            all(s>=p for s,p in zip(shape, self.patch_size)) or _raise(ValueError('patch size > image size'))
            rng = np.random.RandomState(i) if self.same_patches else np.random
            start = (rng.randint(0,1+s-p) for s,p in zip(shape, self.patch_size))
            return tuple(slice(st,st+p) for st,p in zip(start, self.patch_size))
        return tuple(_single_patch(s,i) for s,i in zip(shapes,idx))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = list(self.perm[idx])
        patches = self._random_patches([self.X[k].shape for k in idx], idx)
        X = [self.X[k][sl][self.b] for k,sl in zip(idx,patches)]
        Y = [self.Y[k][sl]         for k,sl in zip(idx,patches)]

        prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            dist      = np.stack([star_dist(lbl,self.n_rays)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist_mask = np.stack([edt_prob(lbl[self.b]) for lbl in Y_cleared])
        else:
            dist      = np.stack([star_dist(lbl,self.n_rays) for lbl in Y])
            dist_mask = prob

        X = np.stack(X)
        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        # subsample wth given grid
        dist_mask = dist_mask[self.ss_grid]
        prob      = prob[self.ss_grid]
        dist      = dist[self.ss_grid]

        return [X,dist_mask], [prob,dist]



class Config(BaseConfig):
    """Configuration for a :class:`StarDist` model.

    TODO: update

    Parameters
    ----------
    n_rays : int
        Number of radial directions for the star-convex polygon.
        Recommended to use a power of 2 (default: 32).
    n_channel_in : int
        Number of channels of given input image (default: 1).
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
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    train_shape_completion : bool
        Train model to predict complete shapes for partially visible objects at image boundary.
    train_completion_crop : int
        If 'train_shape_completion' is set to True, specify number of pixels to crop at boundary of training patches.
        Should be chosen based on (largest) object sizes.
    train_patch_size : (int,int)
        Size of patches to be cropped from provided training images.
    train_dist_loss : str
        Training loss for star-convex polygon distances ('mse' or 'mae').
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
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable.
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable.

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, n_rays=32, n_channel_in=1, grid=(1,1), backbone='unet', **kwargs):
        """See class docstring."""

        super().__init__(axes='YX', n_channel_in=n_channel_in, n_channel_out=1+n_rays)

        # directly set by parameters
        self.n_rays                    = int(n_rays)
        self.grid                      = _normalize_grid(grid,2)
        self.backbone                  = str(backbone).lower()

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
            self.unet_prefix           = ''
            self.net_conv_after_unet   = 128
        else:
            raise ValueError("backbone '%s' not supported." % self.backbone)

        if backend_channels_last():
            self.net_input_shape       = None,None,self.n_channel_in
            self.net_mask_shape        = None,None,1
        else:
            self.net_input_shape       = self.n_channel_in,None,None
            self.net_mask_shape        = 1,None,None

        self.train_shape_completion    = False
        self.train_completion_crop     = 32
        self.train_patch_size          = 256,256
        # self.train_background_reg      = 1e-4 # TODO

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = 1,1
        self.train_epochs              = 100
        self.train_steps_per_epoch     = 400
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 4
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        self.train_reduce_lr        = {'factor': 0.5, 'patience': 10, min_delta_key: 0}

        self.update_parameters(False, **kwargs)



class StarDist(BaseModel):
    """StarDist model.

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

    def __init__(self, config=Config(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        self.config.backbone == 'unet' or _raise(NotImplementedError())

        input_img  = Input(self.config.net_input_shape, name='input')
        if backend_channels_last():
            grid_shape = tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[:-1])) + (1,)
        else:
            grid_shape = (1,) + tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[1:]))
        input_mask = Input(grid_shape, name='dist_mask')

        unet_kwargs = {k[5:]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        # maxpool input image to grid size
        pooled = np.array([1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv2D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding="same", activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling2D(pool)(pooled_img)

        unet        = unet_block(**unet_kwargs)(pooled_img)
        if self.config.net_conv_after_unet > 0:
            unet    = Conv2D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                             name='features', padding='same', activation=self.config.unet_activation)(unet)

        output_prob  = Conv2D(1,                  (1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist  = Conv2D(self.config.n_rays, (1,1), name='dist', padding='same', activation='linear')(unet)
        return Model([input_img,input_mask], [output_prob,output_dist])


    def prepare_for_training(self, optimizer=None):
        """Prepare for neural network training.

        Compiles the model and creates
        `Keras Callbacks <https://keras.io/callbacks/>`_ to be used for training.

        Note that this method will be implicitly called once by :func:`train`
        (with default arguments) if not done so explicitly beforehand.

        Parameters
        ----------
        optimizer : obj or None
            Instance of a `Keras Optimizer <https://keras.io/optimizers/>`_ to be used for training.
            If ``None`` (default), uses ``Adam`` with the learning rate specified in ``config``.

        """
        # TODO: make this exactly the same as in 3D version
        if optimizer is None:
            optimizer = Adam(lr=self.config.train_learning_rate)

        dist_loss = {'mse': masked_loss_mse, 'mae': masked_loss_mae}[self.config.train_dist_loss]
        input_mask = self.keras_model.inputs[1] # second input layer is mask for dist loss
        self.keras_model.compile(optimizer, loss=['binary_crossentropy',dist_loss(input_mask)],
                                            loss_weights = list(self.config.train_loss_weights))

        self.callbacks = []
        if self.basedir is not None:
            self.callbacks += self._checkpoint_callbacks()

            if self.config.train_tensorboard:
                # TODO: CARETensorBoard
                self.callbacks.append(TensorBoard(log_dir=str(self.logdir), write_graph=False))

        if self.config.train_reduce_lr is not None:
            rlrop_params = self.config.train_reduce_lr
            if 'verbose' not in rlrop_params:
                rlrop_params['verbose'] = True
            self.callbacks.append(ReduceLROnPlateau(**rlrop_params))

        self._model_prepared = True


    def train(self, X, Y, validation_data, seed=None, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of input images.
        Y : :class:`numpy.ndarray`
            Array of label masks.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of X,Y validation arrays.
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

        validation_data is not None or _raise(ValueError())
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        b = self.config.train_completion_crop if self.config.train_shape_completion else 0
        div_by = self._axes_div_by(axes)
        [(p-2*b) % d == 0 or _raise(ValueError(
            "'train_patch_size' - 2*'train_completion_crop' must be divisible by {d} along axis '{a}'".format(a=a,d=d) if self.config.train_shape_completion else
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = {
            'n_rays':           self.config.n_rays,
            'batch_size':       self.config.train_batch_size,
            'patch_size':       self.config.train_patch_size,
            'grid':             self.config.grid,
            'shape_completion': self.config.train_shape_completion,
            'b':                self.config.train_completion_crop,
        }

        # TODO: baked validation data -> already done in stardist_public

        X_val, Y_val = validation_data
        data_train = StarDistData(X,     Y,     same_patches=False, **data_kwargs)
        data_val   = StarDistData(X_val, Y_val, same_patches=True,  **data_kwargs)

        history = self.keras_model.fit_generator(generator=data_train, validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history


    def predict(self, img, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, **predict_kwargs):
        """Predict.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as defnoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.

        Returns
        -------
        (:class:`numpy.ndarray`,:class:`numpy.ndarray`)
            Returns the tuple (`prob`, `dist`) of per-pixel object probabilities and star-convex polygon distances.

        """
        if n_tiles is None:
            n_tiles = [1]*img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)
        all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int,n_tiles))

        if axes is None:
            axes = self.config.axes
            assert 'C' in axes
            if img.ndim == len(axes)-1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace('C','')

        axes     = axes_check_and_normalize(axes,img.ndim)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img) # x has axes_net semantics

        channel = axes_dict(axes_net)['C']
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net)-1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace('C',''),grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = StarDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        def predict_direct(tile):
            sh = list(tile.shape); sh[channel] = 1; dummy = np.empty(sh,np.float32)
            prob, dist = self.keras_model.predict([tile[np.newaxis],dummy[np.newaxis]], **predict_kwargs)
            return prob[0], dist[0]

        if np.prod(n_tiles) > 1:
            tiling_axes   = axes_net.replace('C','') # axes eligible for tiling
            x_tiling_axis = tuple(axes_dict(axes_net)[a] for a in tiling_axes) # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            n_tiles = _permute_axes(np.empty(n_tiles,np.bool)).shape
            (all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis) or
                _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))

            sh = [s//grid_dict.get(a,1) for a,s in zip(axes_net,x.shape)]
            sh[channel] = 1;                  prob = np.empty(sh,np.float32)
            sh[channel] = self.config.n_rays; dist = np.empty(sh,np.float32)

            n_block_overlaps = [int(np.ceil(overlap/blocksize)) for overlap, blocksize
                                in zip(axes_net_tile_overlaps, axes_net_div_by)]

            for tile, s_src, s_dst in tqdm(tile_iterator(x, n_tiles, block_sizes=axes_net_div_by, n_block_overlaps=n_block_overlaps),
                                           disable=(not show_tile_progress), total=np.prod(n_tiles)):
                prob_tile, dist_tile = predict_direct(tile)
                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                prob[s_dst] = prob_tile[s_src]
                dist[s_dst] = dist_tile[s_src]

        else:
            prob, dist = predict_direct(x)

        prob = resizer.after(prob, axes_net)
        dist = resizer.after(dist, axes_net)

        prob = np.take(prob,0,axis=channel)
        dist = np.moveaxis(dist,channel,-1)

        return prob, dist


    def _instances_from_prediction(self, img_shape, prob, dist, prob_thresh=0.5, nms_thresh=0.5, return_polygons=False, **nms_kwargs):
        coord = dist_to_coord(dist, grid=self.config.grid)
        points = non_maximum_suppression(coord, prob, grid=self.config.grid,
                                         prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
        labels = polygons_to_label(coord, prob, points, shape=img_shape)
        if return_polygons:
            return labels, coord[points[:,0],points[:,1]], points, prob[points[:,0],points[:,1]]
        else:
            return labels

    def predict_instances(self, img, axes=None, normalizer=None, prob_thresh=0.5, nms_thresh=0.5,
                          return_polygons=False, n_tiles=None, show_tile_progress=True,
                          predict_kwargs=None, nms_kwargs=None):
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        prob, dist = self.predict(img, axes=axes, normalizer=normalizer,
                                  n_tiles=n_tiles, show_tile_progress=show_tile_progress, **predict_kwargs)

        return self._instances_from_prediction(img.shape, prob, dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                                               return_polygons=return_polygons, **nms_kwargs)


    def _axes_div_by(self, query_axes):
        # TODO: different for 3D model / different backbone
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.unet_pool) == len(self.config.grid)
        div_by = dict(zip(
            self.config.axes.replace('C',''),
            tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
        ))
        return tuple(div_by.get(a,1) for a in query_axes)


    def _axes_tile_overlap(self, query_axes):
        # TODO: different for 3D model / different backbone
        self.config.backbone == 'unet' or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.unet_pool) == len(self.config.grid) == len(self.config.unet_kernel_size)
        # TODO: compute this properly when any value of grid > 1
        # all(g==1 for g in self.config.grid) or warnings.warn('FIXME')
        overlap = dict(zip(
            self.config.axes.replace('C',''),
            tuple(tile_overlap(self.config.unet_n_depth + int(np.log2(g)), k, p)
                  for p,k,g in zip(self.config.unet_pool,self.config.unet_kernel_size,self.config.grid))
        ))
        return tuple(overlap.get(a,0) for a in query_axes)


    @property
    def _config_class(self):
        return Config
