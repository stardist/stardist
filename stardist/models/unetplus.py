import numpy as np
from csbdeep.internals.blocks import conv_block2, resnet_block
from csbdeep.utils import _raise, backend_channels_last
import tensorflow as tf


def conv_block(n_filters,
               kernel_size=(3,3),
               strides=(1,1),
               activation="relu",
               batch_norm=False,
               n_conv = 1,
               n_blocks = 1,
                **kwargs):
    if strides is None: strides = (1,)*len(kernel_size)
    assert len(strides)==len(kernel_size) or _raise(ValueError('kernel and pool sizes must match.'))
    ndim = len(kernel_size)
    (ndim in (2,3) ) or _raise(ValueError('block only supports 2d or 3d.'))
    
    def _f_single(inp):
        x = inp
        for i in range(n_conv):
            x = tf.keras.layers.Conv2D(n_filters,
                                       kernel_size,
                                       use_bias= not batch_norm,
                                       strides=strides if i==n_conv-1 else (1,)*ndim,
                                       padding='same',
                                       **kwargs)(x)
            if batch_norm: x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation)(x)
        return x
    
    def f(inp):
        x = inp
        for _ in range(n_blocks):
            x = _f_single(x)
        return x
            
    return f

def _main_block(n_filters,
                kernel_size=(3,3),
                strides=(1,1),
                activation='relu',
                n_conv=2,
                batch_norm=False,
                residual=False,
                bottleneck = False,
                n_blocks=1,
                **kwargs):

    if bottleneck: n_conv += 1
    n_conv >= 2 or _raise(ValueError('required: n_conv >= 2'))
    len(strides) == len(kernel_size) or _raise(ValueError('kernel and pool sizes must match.'))
    ndim = len(kernel_size)
    (ndim in (2,3) ) or _raise(ValueError('block only supports 2d or 3d.'))

    def _f_single(inp):
        x = inp
        for i in range(n_conv):
            bottle = (bottleneck and i == n_conv//2)
            x = conv_block(n_filters if not bottle else n_filters//2,
                           kernel_size if not bottle else (1,)*ndim,
                           n_conv=1,
                           strides=strides if i==0 else (1,)*ndim,
                           batch_norm=batch_norm,
                           activation=activation if i< n_conv-1 else 'linear',
                           **kwargs)(x)

        if residual:
            if max(strides)>1 or n_filters != tf.keras.backend.int_shape(inp)[-1]:
                inp = conv_block(n_filters, (1,)*ndim,
                                 n_conv=1,
                                 strides=strides,
                                 batch_norm=batch_norm,
                                 activation=activation,
                             **kwargs)(inp)
                
                x = tf.keras.layers.Add()([inp, x])
            
        x = tf.keras.layers.Activation(activation)(x)
        return x

    def f(inp):
        x = inp
        for _ in range(n_blocks):
            x = _f_single(x)
        return x

    return f


def conv_basic_block(*args, **kwargs):
    kwargs['bottleneck'] = False
    kwargs['residual']   = False
    return _main_block(*args, **kwargs)

def conv_bottleneck_block(*args, **kwargs):
    kwargs['bottleneck'] = True
    kwargs['residual']   = False
    return _main_block(*args, **kwargs)


def residual_basic_block(*args, **kwargs):
    kwargs['bottleneck'] = False
    kwargs['residual']   = True
    return _main_block(*args, **kwargs)

def residual_bottleneck_block(*args, **kwargs):
    kwargs['bottleneck'] = True
    kwargs['residual']   = True
    return _main_block(*args, **kwargs)

#--------------------------------------------------------------------

def unet_block(
        n_depth=3,
        n_filter_base=32,
        kernel_size=(3,3),
        strides = (2,2),
        block='conv_basic', # 'conv_basic', 'conv_bottleneck,'residual_basic', 'residual_bottleneck'
        n_blocks=2,
        expansion=2,
        activation="relu",
        batch_norm=False):
    """
    Normal Unet
    """

    ndim = len(strides)
    assert len(strides) == len(kernel_size) == ndim

    d_blocks = {
        'conv_basic':conv_basic_block,
        'conv_bottleneck':conv_bottleneck_block,
        'res_basic': residual_basic_block,
        'res_bottleneck':residual_bottleneck_block,
    }

    if not block in d_blocks:
        raise KeyError(f'Unknown block {block}!')
    
    block = d_blocks[block]
    
    pooling    = tf.keras.layers.MaxPooling2D
    upsampling = tf.keras.layers.UpSampling2D
    block_kwargs = dict(
        kernel_initializer="he_normal",
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        activation=activation,
        batch_norm=batch_norm
    )
    
    channel_axis = -1 if backend_channels_last() else 1

    def _f_level(n, inp):

        x = inp

        x = block(n_filters=int(n_filter_base * expansion ** n),**block_kwargs)(x)

        if n<n_depth:
            # the path comming up
            x2 = pooling(strides)(x)
            x2 = _f_level(n+1, x2)
            x2 = upsampling(strides)(x2)

            # concatenate
            x = tf.keras.layers.Concatenate()([x2,x])

            x = tf.keras.layers.Conv2D(int(n_filter_base * expansion ** n),
                                       (1,)*ndim,
                                       padding='same', activation=activation)(x)

        return x
    
    def f(inp):
        return _f_level(0, inp)
        
    return f

def unetplus_block(
        n_depth=3,
        n_filter_base=32,
        kernel_size=(3,3),
        strides = (2,2),
        block='conv_bottleneck', # 'conv_basic', 'residual_basic', 'residual_bottleneck'
        n_blocks=2,
        expansion=1.5,
        multi_heads = False,
        activation="relu",
        batch_norm=False):
    """
    Normal Unet
    """

    ndim = len(strides)
    assert len(strides) == len(kernel_size) == ndim


    d_blocks = {
        'conv_basic':conv_basic_block,
        'conv_bottleneck':conv_bottleneck_block,
        'res_basic': residual_basic_block,
        'res_bottleneck':residual_bottleneck_block,
    }

    if not block in d_blocks:
        raise KeyError(f'Unknown block {block}!')

    block = d_blocks[block]
    
    pooling    = tf.keras.layers.MaxPooling2D
    upsampling = tf.keras.layers.UpSampling2D
    block_kwargs = dict(
        kernel_initializer="he_normal",
        n_blocks=n_blocks,
        kernel_size=kernel_size,
        activation=activation,
        batch_norm=batch_norm
    )
    
    channel_axis = -1 if backend_channels_last() else 1

    def _f_level(n, inp):

        x = block(n_filters=int(n_filter_base * expansion ** n),**block_kwargs)(inp)

        
        if n<n_depth:
            # the lower part (recursive)
            x2 = pooling(strides)(x)
            x2_inter, heads = _f_level(n+1, x2)
            x2_inter = tuple(upsampling(strides)(_x) for _x in x2_inter)
        else:
            heads = []
            
        x_inter = [x] 
        for i in range(n_depth-n):
            print(i)
            
            x = block(n_filters=int(n_filter_base * expansion ** n),**block_kwargs)(x)
            
            if i==0:
                inter_concat = [x_inter[0]]
            else:
                inter_concat = x_inter

            if n<n_depth:
                inter_concat = [x2_inter[i]] + inter_concat
                
            x = tf.keras.layers.Concatenate()(inter_concat + [x])
            
            x = tf.keras.layers.Conv2D(int(n_filter_base * expansion ** n),
                                       (1,)*ndim,
                                       padding='same', activation=activation)(x)
            
            x_inter.append(x)

        heads = [x] + heads
        return x_inter, heads
    
    def f(inp):
        x_inter, heads =  _f_level(0, inp)
        if multi_heads:
            return heads
        else:
            return heads[0]
        
    return f


# ----------------------


def unet_model(input_shape,
               last_activation='linear',
               n_classes=1,
               n_depth=3,
               n_filter_base=32,
               kernel_size=(3,3),
               strides = (2,2),
               block='conv_basic',
               n_blocks=2,
               activation="relu",
               batch_norm=False):
    
    inp = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(n_filter_base,(5,5), padding='same',activation=activation)(inp)

    feat = unet_block(n_depth=n_depth,
                      n_filter_base=n_filter_base,
                      kernel_size=kernel_size,
                      strides=strides,
                      block=block,
                          n_blocks=n_blocks,
                      activation=activation)(x)

    out = tf.keras.layers.Conv2D(n_classes,(1,1), padding='same',activation=last_activation)(feat)

    model = tf.keras.models.Model(inp, out)
    return model
    

def unetplus_model(input_shape,
                   last_activation='linear',
                   n_classes=1,
                   n_depth=3,
                   n_filter_base=32,
                   kernel_size=(3,3),
                   strides = (2,2),
                   block='conv_basic',
                   n_blocks=2,
                   expansion=2,
                   multi_heads = False,
                   activation="relu",
                   batch_norm=False):
    
    inp = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(n_filter_base,(5,5), padding='same',activation=activation)(inp)

    feat = unetplus_block(n_depth=n_depth,
                          n_filter_base=n_filter_base,
                          kernel_size=kernel_size,
                          strides=strides,
                          block=block,
                          n_blocks=n_blocks,
                          expansion=expansion,
                          multi_heads=multi_heads,  
                          activation=activation)(x)

    if multi_heads:
        out = tuple(tf.keras.layers.Conv2D(n_classes,(1,1), padding='same',activation=last_activation)(f) for f in feat)
    else:
        out = tf.keras.layers.Conv2D(n_classes,(1,1), padding='same',activation=last_activation)(feat)

    model = tf.keras.models.Model(inp, out)
    return model





if __name__ == '__main__':

    model = unetplus_model((64,64,1), 'sigmoid', multi_heads=True)
