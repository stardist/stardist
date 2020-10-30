from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import warnings
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage
from csbdeep.internals.blocks import conv_block2, conv_block3

keras = keras_import()
K = keras_import('backend')
Input, Dense, Multiply , Activation, Add, Concatenate    = keras_import('layers', 'Input', 'Dense','Multiply','Activation','Add','Concatenate')
Conv2D, MaxPooling2D,GlobalAveragePooling2D, UpSampling2D = keras_import('layers','Conv2D', 'MaxPooling2D','GlobalAveragePooling2D', 'UpSampling2D')
Conv3D, MaxPooling2D,GlobalAveragePooling3D, UpSampling3D = keras_import('layers','Conv3D', 'MaxPooling2D','GlobalAveragePooling3D','UpSampling3D')


def se_block2(inp, out, n_channel, ratio=8):
    y = GlobalAveragePooling2D()(inp)
    y = Dense(n_channel//ratio, activation='relu')(y)
    y = Dense(n_channel, activation='sigmoid')(y)
    return Multiply()([out, y])


def se_block3(inp, out, n_channel, ratio=8):
    y = GlobalAveragePooling3D()(inp)
    y = Dense(n_channel//ratio, activation='relu')(y)
    y = Dense(n_channel, activation='sigmoid')(y)
    return Multiply()([out, y])


def resnetSE_block(n_filter, kernel_size=(3,3), pool=(1,1), n_conv_per_block=2,
                 batch_norm=False, kernel_initializer='he_normal', activation='relu'):
    """ Squeeze and Excite 
    https://arxiv.org/abs/1709.01507
    """

    n_conv_per_block >= 2 or _raise(ValueError('required: n_conv_per_block >= 2'))
    len(pool) == len(kernel_size) or _raise(ValueError('kernel and pool sizes must match.'))
    n_dim = len(kernel_size)
    n_dim in (2,3) or _raise(ValueError('resnet_block only 2d or 3d.'))

    conv_layer = Conv2D if n_dim == 2 else Conv3D
    se_block = se_block2 if n_dim == 2 else se_block3
    
    conv_kwargs = dict (
        padding            = 'same',
        use_bias           = not batch_norm,
        kernel_initializer = kernel_initializer,
    )
    channel_axis = -1 if backend_channels_last() else 1

    def f(inp):
        x = conv_layer(n_filter, kernel_size, strides=pool, **conv_kwargs)(inp)
        if batch_norm:
            x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(activation)(x)

        for _ in range(n_conv_per_block-2):
            x = conv_layer(n_filter, kernel_size, **conv_kwargs)(x)
            if batch_norm:
                x = BatchNormalization(axis=channel_axis)(x)
            x = Activation(activation)(x)

        x = conv_layer(n_filter, kernel_size, **conv_kwargs)(x)
        if batch_norm:
            x = BatchNormalization(axis=channel_axis)(x)

        if any(p!=1 for p in pool) or n_filter != K.int_shape(inp)[-1]:
            inp = conv_layer(n_filter, (1,)*n_dim, strides=pool, **conv_kwargs)(inp)

        x = se_block(x,x,n_filter, ratio=min(n_filter, 8))
        
        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f

def unetSE_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D
    se_block   = se_block2 if n_dim == 2 else se_block3

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            in_layer = layer
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            layer = se_block(in_layer,layer, n_filter_base* 2 ** n, ratio=min(n_filter_base* 2 ** n, 8))
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** n_depth, *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), *kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, *kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func



