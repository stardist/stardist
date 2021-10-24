"""
Feature pyramid network
"""

from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
from csbdeep.utils import _raise, backend_channels_last
from csbdeep.utils.tf import keras_import
K = keras_import('backend')
Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, Cropping2D, Cropping3D, Concatenate, Add, Dropout, Activation, BatchNormalization = \
    keras_import('layers', 'Conv2D', 'MaxPooling2D', 'UpSampling2D', 'Conv3D', 'MaxPooling3D', 'UpSampling3D', 'Cropping2D', 'Cropping3D', 'Concatenate', 'Add', 'Dropout', 'Activation', 'BatchNormalization')

from csbdeep.internals.blocks import conv_block2, conv_block3
import warnings


def resnet_block(n_filter, kernel_size=(3,3), pool=(1,1), n_conv_per_block=2,
                 batch_norm=False, kernel_initializer='he_normal', activation='relu'):

    n_conv_per_block >= 2 or _raise(ValueError('required: n_conv_per_block >= 2'))
    len(pool) == len(kernel_size) or _raise(ValueError('kernel and pool sizes must match.'))
    n_dim = len(kernel_size)
    n_dim in (2,3) or _raise(ValueError('resnet_block only 2d or 3d.'))

    conv_layer = Conv2D if n_dim == 2 else Conv3D
    conv_kwargs = dict (
        padding            = 'same',
        kernel_initializer = kernel_initializer,
    )
    channel_axis = -1 if backend_channels_last() else 1

    def f(inp):
        # first conv to prepare filter sizes and strides...
        x = conv_layer(n_filter, kernel_size, strides=pool, use_bias=not batch_norm,**conv_kwargs)(inp)
        if batch_norm: x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(activation)(x)

        # extra middle conv if n_conv_per_block>2
        for _ in range(n_conv_per_block-2):
            x = conv_layer(n_filter, kernel_size, use_bias=not batch_norm, **conv_kwargs)(x)
            if batch_norm: x = BatchNormalization(axis=channel_axis)(x)
            x = Activation(activation)(x)

        # last conv with no activation for residual addition
        x = conv_layer(n_filter, kernel_size, use_bias=not batch_norm, **conv_kwargs)(x)
        if batch_norm: x = BatchNormalization(axis=channel_axis)(x)

        # transform shortcut input if not compatible...
        if any(tuple(p!=1 for p in pool)) or n_filter != K.int_shape(inp)[-1]:
            inp = conv_layer(n_filter, (1,)*n_dim, strides=pool, **conv_kwargs)(inp)

        x = Add()([inp, x])
        x = Activation(activation)(x)
        return x

    return f




def fpn_block(n_depth=3,
              n_filter_base=32,
              kernel_size=(3,3),
              head_filters=32,
              multi_head=False,
              n_conv_per_depth=2,
              activation="elu",
              batch_norm=True,
              dropout=0.0,
              last_activation='elu',
              pool=(2,2),
              kernel_init="he_normal",
              prefix=''):
    """
    Returns multiscale resolution maps of size head_filters 

    If multi_head=False all these are concatenated

    Highly advised to use batch_norm due to the identity part taking over !

    """

    if not batch_norm:
        warnings.warn("BatchNormalisation disabled in Feature Pyramid Network!")

        
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

    # expansion factor of filters 
    alpha = 1.66
    
    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            n_filt = int(n_filter_base * alpha ** n)
            for _ in range(n_conv_per_depth):
                layer = resnet_block(n_filt,
                                     kernel_size, pool=(1,)*len(kernel_size),
                                     n_conv_per_block=2,
                                     activation=activation,
                                     batch_norm=batch_norm)(layer)
            
            skip_layers.append(layer)
            layer = pooling(pool)(layer)

            
        heads = []
        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            n_filt = int(n_filter_base * alpha ** max(0, n_depth - 1))
            
            for _ in range(n_conv_per_depth):
                layer = resnet_block(n_filt,
                                 kernel_size, pool=(1,)*len(kernel_size),
                                 n_conv_per_block=2,activation=activation,
                                 batch_norm=batch_norm)(layer)
            
            layer = upsampling(pool, interpolation='bilinear')(layer)
            
            skip = conv_block(n_filt, *((1,)*n_dim),
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation="linear",
                                   batch_norm=batch_norm)(skip_layers[n])
            
            layer = Add()([layer, skip])

            head=layer

            head = upsampling(tuple(p**n for p in pool), interpolation='bilinear')(head)

            head = conv_block(head_filters, *kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=last_activation,
                               batch_norm=batch_norm)(head)

            heads.append(head)

        if multi_head:
            final = heads
        else:
            final = Concatenate()(heads)

        #     final = conv_block(n_filter_base, *kernel_size,
        #                    dropout=dropout,
        #                    init=kernel_init,
        #                    activation=last_activation,
        #                    batch_norm=batch_norm, name=_name("last"))(layer)

        return final

    return _func



if __name__ == '__main__':

    from csbdeep.internals.blocks import unet_block
    
    import tensorflow as tf
    
    tf.random.set_seed(42)
    np.random.seed(42)

    inp = tf.keras.layers.Input((None,None, 1))

    features = fpn_block(n_filter_base=64, n_depth=4, n_conv_per_depth=2, head_filters=64)(inp)

    out = tf.keras.layers.Conv2D(1,(1,1), padding='same', activation='sigmoid')(features)

    model = tf.keras.models.Model(inp, out)
    
    model.summary()
    
    # inp = tf.keras.layers.Input((None,None, 1))

    # features = fpn_block(n_filter_base=32, n_depth=4, n_conv_per_depth=2, head_filters=32)(inp)

    # out = tf.keras.layers.Conv2D(1,(1,1), padding='same', activation='sigmoid')(features)

    # model = tf.keras.models.Model(inp, out)
    
    # model.summary()
    
    # inp = tf.keras.layers.Input((None,None, 1))

    # features = fpn_block(n_filter_base=32, n_depth=4, n_conv_per_depth=2, head_filters=32)(inp)

    # out = tf.keras.layers.Conv2D(1,(1,1), padding='same', activation='sigmoid')(features)

    # model_fpn = tf.keras.models.Model(inp, out)
    

    # features = unet_block(n_depth=4, last_activation='relu')(inp)
    # out = tf.keras.layers.Conv2D(1,(1,1), padding='same',
    #                               activation='sigmoid')(features)
    # model_unet = tf.keras.models.Model(inp, out)

    # models = dict(unet=model_unet, fpn=model_fpn)
    

    # from stardist.data import test_image_nuclei_2d
    # x, y = test_image_nuclei_2d(return_mask=True)
    # x = (x /255).astype(np.float32)
    # y = (y>0).astype(np.float32)
    # x = np.repeat(np.expand_dims(x,0),16, axis=0)
    # y = np.repeat(np.expand_dims(y,0),16, axis=0)

    # hist = dict()
    
    # for k,model in models.items():
    #     print('+'*100)
    #     print(k)
    #     model.compile(loss='mse', optimizer= tf.keras.optimizers.Adam(lr=3e-4))
    #     hist[k] = model.fit(x,y, epochs=50, batch_size=1)


        
    # def lay(m,n):
    #     f = tf.keras.backend.function(m.input, m.layers[n].output)
    #     return f(x[:1])[0]

    # def show(layer=-1, **kwargs):
    #     for i,(k,model) in enumerate(models.items()):
    #         act = lay(model, layer)
    #         plt.subplot(1,len(models.keys())+1,i+1)
    #         plt.imshow(np.mean(act,axis=-1),**kwargs)
    #     plt.subplot(1,len(models.keys())+1,len(models.keys())+1)
    #     plt.cla()
    #     for k,h in hist.items():
    #         plt.plot(h.history['loss'], label = k)
    #         plt.gca().set_yscale('log')
    #     plt.legend()

    # import matplotlib.pyplot as plt 
    # plt.ion()
    # show(clim=(0,1))
    # plt.show()
