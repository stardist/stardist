from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import warnings
from csbdeep.utils import _raise, backend_channels_last, axes_check_and_normalize, axes_dict
from csbdeep.utils.tf import keras_import, IS_TF_1, CARETensorBoard, CARETensorBoardImage

keras = keras_import()
K = keras_import('backend')
Input, Dense, Multiply , Activation,Add      = keras_import('layers', 'Input', 'Dense','Multiply','Activation','Add')
Conv2D, MaxPooling2D,GlobalAveragePooling2D = keras_import('layers','Conv2D', 'MaxPooling2D','GlobalAveragePooling2D')
Conv3D, MaxPooling2D,GlobalAveragePooling3D = keras_import('layers','Conv3D', 'MaxPooling2D','GlobalAveragePooling3D')

def se_block2(x, n_channel, ratio=16):
    y = GlobalAveragePooling2D()(x)
    y = Dense(n_channel//ratio, activation='relu')(y)
    y = Dense(n_channel, activation='sigmoid')(y)
    return Multiply()([x, y])

def se_block3(x, n_channel, ratio=16):
    y = GlobalAveragePooling3D()(x)
    y = Dense(n_channel//ratio, activation='relu')(y)
    y = Dense(n_channel, activation='sigmoid')(y)
    return Multiply()([x, y])

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

        x = se_block(x,n_filter, ratio=min(n_filter, 16))
        
        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f
