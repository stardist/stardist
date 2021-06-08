from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
from csbdeep.utils import _raise, backend_channels_last

from csbdeep.utils.tf import keras_import
K = keras_import('backend')
Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Cropping2D, Cropping3D, Concatenate, Add, Dropout, Activation, BatchNormalization = \
    keras_import('layers', 'Conv2D', 'MaxPooling2D', 'UpSampling2D', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Cropping2D', 'Cropping3D', 'Concatenate', 'Add', 'Dropout', 'Activation', 'BatchNormalization')

from csbdeep.internals.blocks import conv_block2, conv_block3, resnet_block



def fpn_block(n_depth=3,
              n_filter_base=64,
              kernel_size=(3,3),
              pyramid_filters=128,
              n_conv_per_depth=2,
              activation="relu",
              batch_norm=False,
              dropout=0.0,
              last_activation=None,
              pool=(2,2),
              kernel_init="he_normal",
              prefix=''):

    if len(pool) != len(kernel_size):
        raise ValueError('kernel and pool sizes must match.')
    n_dim = len(kernel_size)
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D if n_dim == 2 else UpSampling3D

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
            layer = resnet_block(n_filter_base * 2 ** n,
                                 kernel_size, pool=(1,)*len(kernel_size),
                                 n_conv_per_block=n_conv_per_depth,activation=activation,
                                 batch_norm=batch_norm)(layer)
            
            skip_layers.append(layer)
            layer = pooling(pool)(layer)

        heads = []
        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = resnet_block(n_filter_base * 2 ** max(0, n_depth - 1),
                                 kernel_size, pool=(1,)*len(kernel_size),
                                 n_conv_per_block=n_conv_per_depth,activation=activation,
                                 batch_norm=batch_norm)(layer)
            
            layer = conv_block(pyramid_filters, *((1,)*n_dim),
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation="linear",
                                   batch_norm=batch_norm)(layer)
            layer = upsampling(pool)(layer)
            
            skip = conv_block(pyramid_filters, *((1,)*n_dim),
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation="linear",
                                   batch_norm=batch_norm)(skip_layers[n])
            
            layer = Add()([layer, skip])

            head = conv_block(pyramid_filters, *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm)(layer)

            head = upsampling(tuple(p**n for p in pool))(head)

            heads.append(head)

            
        layer = Concatenate()(heads)

        final = conv_block(n_filter_base, *kernel_size,
                           dropout=dropout,
                           init=kernel_init,
                           activation=last_activation,
                           batch_norm=batch_norm, name=_name("last"))(layer)

        return final

    return _func



if __name__ == '__main__':


    Input = keras_import('layers', 'Input')
    Model = keras_import('models', 'Model')


    inp = Input((128,128,1))
    out = fpn_block(3)(inp)
    model = Model(inp, out)

    model.summary()

    print(model.output_shape)
