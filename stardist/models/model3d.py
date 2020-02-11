from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import warnings
import math
from tqdm import tqdm

from distutils.version import LooseVersion
import keras
import keras.backend as K
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Concatenate
from keras.models import Model

from csbdeep.models import BaseConfig
from csbdeep.internals.blocks import conv_block3, unet_block, resnet_block
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import CARETensorBoard

from .base import StarDistBase, StarDistDataBase
from .sample_patches import sample_patches
from ..utils import edt_prob, _normalize_grid
from ..matching import relabel_sequential
from ..geometry import star_dist3D, polyhedron_to_label
from ..rays3d import Rays_GoldenSpiral, rays_from_json
from ..nms import non_maximum_suppression_3d



class StarDistData3D(StarDistDataBase):

    def __init__(self, X, Y, batch_size, rays, patch_size=(128,128,128), grid=(1,1,1), anisotropy=None, augmenter=None, foreground_prob=0, **kwargs):
        # TODO: support shape completion as in 2D?

        super().__init__(X=X, Y=Y, n_rays=len(rays), grid=grid,
                         batch_size=batch_size, patch_size=patch_size,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.rays = rays
        self.anisotropy = anisotropy
        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'

        # re-use arrays
        if self.batch_size > 1:
            self.out_X = np.empty((self.batch_size,)+tuple(self.patch_size)+(() if self.n_channel is None else (self.n_channel,)), dtype=np.float32)
            self.out_edt_prob = np.empty((self.batch_size,)+tuple(self.patch_size), dtype=np.float32)
            self.out_star_dist3D = np.empty((self.batch_size,)+tuple(self.patch_size)+(len(self.rays),), dtype=np.float32)


    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = list(self.perm[idx])

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

        tmp = [edt_prob(lbl, anisotropy=self.anisotropy) for lbl in Y]
        if len(Y) == 1:
            prob = tmp[0][np.newaxis]
        else:
            prob = np.stack(tmp, out=self.out_edt_prob[:len(Y)])

        tmp = [star_dist3D(lbl, self.rays, mode=self.sd_mode) for lbl in Y]
        if len(Y) == 1:
            dist = tmp[0][np.newaxis]
        else:
            dist = np.stack(tmp, out=self.out_star_dist3D[:len(Y)])

        prob = dist_mask = np.expand_dims(prob, -1)

        # subsample wth given grid
        dist_mask = dist_mask[self.ss_grid]
        prob      = prob[self.ss_grid]
        dist      = dist[self.ss_grid]

        return [X,dist_mask], [prob,dist]



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

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes='ZYX', rays=None, n_channel_in=1, grid=(1,1,1), anisotropy=None, backbone='resnet', **kwargs):

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

        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = 1,0.2
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
            raise NotImplementedError()


    def _build_unet(self):
        assert self.config.backbone == 'unet'

        input_img = Input(self.config.net_input_shape, name='input')
        if backend_channels_last():
            grid_shape = tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[:-1])) + (1,)
        else:
            grid_shape = (1,) + tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[1:]))
        input_mask = Input(grid_shape, name='dist_mask')

        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        # maxpool input image to grid size
        pooled = np.array([1,1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv3D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding="same", activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling3D(pool)(pooled_img)

        unet     = unet_block(**unet_kwargs)(pooled_img)
        if self.config.net_conv_after_unet > 0:
            unet = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet)

        output_prob = Conv3D(1,                  (1,1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(unet)
        return Model([input_img,input_mask], [output_prob,output_dist])


    def _build_resnet(self):
        assert self.config.backbone == 'resnet'

        input_img = Input(self.config.net_input_shape, name='input')
        if backend_channels_last():
            grid_shape = tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[:-1])) + (1,)
        else:
            grid_shape = (1,) + tuple(n//g if n is not None else None for g,n in zip(self.config.grid, self.config.net_mask_shape[1:]))
        input_mask = Input(grid_shape, name='dist_mask')

        n_filter = self.config.resnet_n_filter_base
        resnet_kwargs = dict (
            kernel_size        = self.config.resnet_kernel_size,
            n_conv_per_block   = self.config.resnet_n_conv_per_block,
            batch_norm         = self.config.resnet_batch_norm,
            kernel_initializer = self.config.resnet_kernel_init,
            activation         = self.config.resnet_activation,
        )

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

        if self.config.net_conv_after_resnet > 0:
            layer = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                           name='features', padding='same', activation=self.config.resnet_activation)(layer)

        output_prob = Conv3D(1,                  (1,1,1), name='prob', padding='same', activation='sigmoid')(layer)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(layer)
        return Model([input_img,input_mask], [output_prob,output_dist])


    def train(self, X,Y, validation_data, augmenter=None, seed=None, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of input images.
        Y : :class:`numpy.ndarray`
            Array of label masks.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of X,Y validation arrays.
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

        validation_data is not None or _raise(ValueError())
        ((isinstance(validation_data,(list,tuple)) and len(validation_data)==2)
            or _raise(ValueError('validation_data must be a pair of numpy arrays')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        div_by = self._axes_div_by(axes)
        [p % d == 0 or _raise(ValueError(
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            rays            = rays_from_json(self.config.rays_json),
            grid            = self.config.grid,
            patch_size      = self.config.train_patch_size,
            anisotropy      = self.config.anisotropy,
            use_gpu         = self.config.use_gpu,
            foreground_prob = self.config.train_foreground_only,
        )

        # generate validation data and store in numpy arrays
        # data_val = StarDistData3D(*validation_data, batch_size=1, augment=False, **data_kwargs)
        _data_val = StarDistData3D(*validation_data, batch_size=1, **data_kwargs)
        n_data_val = len(_data_val)
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        ids = tuple(np.random.choice(n_data_val, size=n_take, replace=(n_take > n_data_val)))
        Xv, Mv, Pv, Dv = [None]*n_take, [None]*n_take, [None]*n_take, [None]*n_take
        for i,k in enumerate(ids):
            (Xv[i],Mv[i]),(Pv[i],Dv[i]) = _data_val[k]
        Xv, Mv, Pv, Dv = np.concatenate(Xv,axis=0), np.concatenate(Mv,axis=0), np.concatenate(Pv,axis=0), np.concatenate(Dv,axis=0)
        data_val = [[Xv,Mv],[Pv,Dv]]

        data_train = StarDistData3D(X, Y, batch_size=self.config.train_batch_size, augmenter=augmenter, **data_kwargs)

        for cb in self.callbacks:
            if isinstance(cb,CARETensorBoard):
                # only show middle slice of 3D inputs/outputs
                cb.input_slices, cb.output_slices = [[slice(None)]*5,[slice(None)]*5], [[slice(None)]*5,[slice(None)]*5]
                i = axes_dict(self.config.axes)['Z']
                _n_in  = _data_val.patch_size[i] // 2
                _n_out = _data_val.patch_size[i] // (2 * (self.config.grid[i] if self.config.grid is not None else 1))
                cb.input_slices[0][1+i] = _n_in
                cb.input_slices[1][1+i] = _n_out
                cb.output_slices[0][1+i] = _n_out
                cb.output_slices[1][1+i] = _n_out
                # show dist for three rays
                _n = min(3, self.config.n_rays)
                cb.output_slices[1][1+axes_dict(self.config.axes)['C']] = slice(0,(self.config.n_rays//_n)*_n,self.config.n_rays//_n)

        history = self.keras_model.fit_generator(generator=data_train, validation_data=data_val,
                                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                 callbacks=self.callbacks, verbose=1)
        self._training_finished()

        return history


    def _instances_from_prediction(self, img_shape, prob, dist, prob_thresh=None, nms_thresh=None, overlap_label = None, **nms_kwargs):
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)
        points, probi, disti = non_maximum_suppression_3d(dist, prob, rays, grid=self.config.grid,
                                                          prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")
        labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label = overlap_label, verbose=verbose)

        # map the overlap_label to something positive and back
        # (as relabel_sequential doesn't like negative values)
        if overlap_label is not None and overlap_label<0:
            overlap_mask = (labels == overlap_label)
            overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
            labels[overlap_mask] = overlap_label2
            labels, fwd, bwd = relabel_sequential(labels)
            labels[labels == fwd[overlap_label2]] = overlap_label
        else:
            labels, _,_ = relabel_sequential(labels)

        # TODO: convert to polyhedra faces?
        return labels, dict(dist=disti, points=points, prob=probi, rays=rays)


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


    # def _axes_tile_overlap(self, query_axes):
    #     # TODO: correct?
    #     if self.config.backbone == "unet":
    #         query_axes = axes_check_and_normalize(query_axes)
    #         assert len(self.config.unet_pool) == len(self.config.grid) == len(self.config.unet_kernel_size)
    #         # TODO: compute this properly when any value of grid > 1
    #         # all(g==1 for g in self.config.grid) or warnings.warn('FIXME')
    #         overlap = dict(zip(
    #             self.config.axes.replace('C',''),
    #             tuple(tile_overlap(self.config.unet_n_depth + int(np.log2(g)), k, p)
    #                   for p,k,g in zip(self.config.unet_pool,self.config.unet_kernel_size,self.config.grid))
    #         ))
    #         return tuple(overlap.get(a,0) for a in query_axes)
    #     elif self.config.backbone == "resnet":
    #         # TODO: compute properly?
    #         return tuple(0 if a == 'C' else 32 for a in query_axes)
    #     else:
    #         raise NotImplementedError()


    @property
    def _config_class(self):
        return Config3D
