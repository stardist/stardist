from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import os
import json
import numpy as np
import argparse
import warnings
import datetime
import math
from six import iteritems
from tqdm import tqdm
from warnings import warn

from distutils.version import LooseVersion
import keras
import keras.backend as K
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Add, Concatenate
from keras.models import Model

from csbdeep.models import BaseConfig, BaseModel
from csbdeep.internals.blocks import conv_block3, unet_block, resnet_block
from csbdeep.internals.predict import tile_iterator, tile_overlap
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import CARETensorBoard
from csbdeep.data import sample_patches_from_multiple_stacks

from .base import StarDistBase, StarDistDataBase, StarDistPadAndCropResizer
from ..utils import edt_prob, _normalize_grid, _is_power_of_2, calculate_extents, relabel_sequential
from ..geometry.three_d import star_dist3D, polyhedron_to_label
from ..rays3d import Rays_GoldenSpiral, rays_from_json
from ..nms import non_maximum_suppression_3d



class StarDistData3D(StarDistDataBase):
    def __init__(self, X, Y, batch_size, rays, patch_size=(128,128,128), grid=(1,1,1), anisotropy=None, **kwargs):

        X = [x.astype(np.float32, copy=False) for x in X]
        super().__init__(X=X, Y=Y, n_rays=len(rays), grid=grid,
                         batch_size=batch_size, patch_size=patch_size,
                         **kwargs)

        self.rays = rays
        self.anisotropy = anisotropy
        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'

        # re-use arrays
        if self.batch_size > 1:
            self.out_X = np.empty((self.batch_size,)+tuple(self.patch_size), dtype=np.float32)
            self.out_edt_prob = np.empty((self.batch_size,)+tuple(self.patch_size), dtype=np.float32)
            self.out_star_dist3D = np.empty((self.batch_size,)+tuple(self.patch_size)+(len(self.rays),), dtype=np.float32)


    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = list(self.perm[idx])

        arrays = [sample_patches_from_multiple_stacks((self.X[k],self.Y[k]),
                                                      patch_size=self.patch_size, n_samples=1,
                                                      patch_filter=self.no_background_patches_cached(k)) for k in idx]
        X, Y = list(zip(*[(x[0],y[0]) for x,y in arrays]))

        # TODO: apply augmentation here

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

    TODO: update

    Parameters
    ----------
    rays : Rays_Base
        ray factory baseclass

    grid : (int,int,int)

    n_channel_in : int
        Number of channels of given input image (default: 1).

    kwargs : dict
        Overwrite (or add) configuration attributes (see below).


    Attributes
    ----------
    unet_n_depth : int
        Number of U-Net resolution levels (down/up-sampling layers).
    unet_kernel_size : (int,int, int)
        Convolution kernel size for all (U-Net) convolution layers.
    unet_pool : (int,int, int)
        Maxpooling  size for all (U-Net) convolution layers.
    unet_n_filter_base : int
        Number of convolution kernels (feature channels) for first U-Net layer.
        Doubled after each down-sampling layer.
    unet_dilation_rates : tuple or None
        dilations rates inside of Unet
    net_conv_after_unet : int
        Number of filters of the extra convolution layer after U-Net (0 to disable).
    sparse_labeling: False
        set to True if GT instance labeling is only sparse
    train_patch_size : (int,int, int)
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

    def __init__(self, axes='ZYX', rays=Rays_GoldenSpiral(96), n_channel_in=1, grid=(1,1,1), anisotropy=None, backbone='resnet', ray_factory=Rays_GoldenSpiral, **kwargs):

        super().__init__(axes=axes, n_channel_in=n_channel_in, n_channel_out=1+len(rays))

        # directly set by parameters
        self.n_rays                    = len(rays)
        self.grid                      = _normalize_grid(grid,3)
        self.anisotropy                = anisotropy if anisotropy is None else tuple(anisotropy)
        self.backbone                  = str(backbone).lower()
        self.rays_json                 = rays.to_json()

        # default config (can be overwritten by kwargs below)
        if self.backbone == 'unet':
            self.unet_n_depth          = 2
            self.unet_kernel_size      = 3,3,3
            self.unet_n_filter_base    = 32
            self.unet_n_conv_per_depth = 2
            self.unet_pool             = 2,2,2
            self.unet_activation       = 'relu'
            self.unet_last_activation  = 'relu'
            self.unet_batch_norm       = False
            self.unet_dropout          = 0.0
            self.unet_prefix           = ''
            self.net_conv_after_unet   = 128
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

        # TODO: good default params?
        self.train_dist_loss           = 'mae'
        self.train_loss_weights        = 1,1
        self.train_epochs              = 200
        self.train_steps_per_epoch     = 100
        self.train_learning_rate       = 0.0003
        self.train_batch_size          = 1
        self.train_tensorboard         = True
        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if LooseVersion(keras.__version__)<=LooseVersion('2.1.5') else 'min_delta'
        self.train_reduce_lr           = {'factor': 0.5, 'patience': 40, min_delta_key: 0}

        self.update_parameters(False, **kwargs)




