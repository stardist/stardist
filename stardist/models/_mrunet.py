"""
Ibtehaz, Nabil, and M. Sohel Rahman. "MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation." Neural Networks 121 (2020): 74-87.

Code adapted to tf 2.0 from https://github.com/nibtehaz/MultiResUNet/blob/master/MultiResUNet.py

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import ELU, LeakyReLU


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', batch_norm=False, name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=not batch_norm)(x)
    
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)

    if not activation is None:
        x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row=2, num_col=2, padding='same', strides=(2, 2), batch_norm=False, name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    # x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)

    x = UpSampling2D(strides, interpolation='bilinear')(x)
    x = Conv2D(filters, (3, 3), padding=padding, activation='relu', use_bias=not batch_norm)(x)

    if batch_norm:
        x = BatchNormalization(axis=-1)(x)
    
    return x


def multi_res_block(U, inp, alpha = 1.67, batch_norm=False):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None,
                         batch_norm=batch_norm, 
                         padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3,3,
                        activation='relu', padding='same', batch_norm=batch_norm)

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3,3,
                        activation='relu', padding='same', batch_norm=batch_norm)

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3,3,
                        activation='relu', padding='same', batch_norm=batch_norm)

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)

    out = add([shortcut, out])
    out = Activation('relu')(out)

    return out


def res_path(filters, length, inp, batch_norm=False):
    '''
    res_path
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of res_path
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same', batch_norm=batch_norm)

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same', batch_norm=batch_norm)

    out = add([shortcut, out])
    out = Activation('relu')(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same', batch_norm=batch_norm)

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same', batch_norm=batch_norm)

        out = add([shortcut, out])
        out = Activation('relu')(out)

    return out


def mrunet_block(n_filters=32, batch_norm=False):

    def f(x):

        mresblock1 = multi_res_block(32, x, batch_norm=batch_norm)
        pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
        mresblock1 = res_path(32, 4, mresblock1, batch_norm=batch_norm)

        mresblock2 = multi_res_block(32*2, pool1, batch_norm=batch_norm)
        pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
        mresblock2 = res_path(32*2, 3, mresblock2, batch_norm=batch_norm)

        mresblock3 = multi_res_block(32*4, pool2, batch_norm=batch_norm)
        pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
        mresblock3 = res_path(32*4, 2, mresblock3, batch_norm=batch_norm)

        mresblock4 = multi_res_block(32*8, pool3, batch_norm=batch_norm)
        pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
        mresblock4 = res_path(32*8, 1, mresblock4, batch_norm=batch_norm)

        mresblock5 = multi_res_block(32*16, pool4, batch_norm=batch_norm)

                
        up6 = concatenate([trans_conv2d_bn(mresblock5, 32*8, strides=(2, 2), batch_norm=batch_norm),
                           mresblock4], axis=3)
        
        mresblock6 = multi_res_block(32*8, up6)

        up7 = concatenate([trans_conv2d_bn(mresblock6, 32*4, strides=(2, 2), batch_norm=batch_norm),
                           mresblock3], axis=3)
        
        mresblock7 = multi_res_block(32*4, up7)

        up8 = concatenate([trans_conv2d_bn(mresblock7, 32*2,  strides=(2, 2), batch_norm=batch_norm),
                           mresblock2], axis=3)
        
        
        mresblock8 = multi_res_block(32*2, up8)

        up9 = concatenate([trans_conv2d_bn(mresblock8, 32, strides=(2, 2), batch_norm=batch_norm),
                           mresblock1], axis=3)
        
        
        mresblock9 = multi_res_block(32, up9, batch_norm=batch_norm)

        return mresblock9


    return f
   


if __name__ == '__main__':


    tf.random.set_seed(42)
    np.random.seed(42)
    
    inp = Input((None,None, 1))

    features = mrunet_block(batch_norm=True)(inp)

    out = tf.keras.layers.Conv2D(1,(1,1), padding='same', activation='sigmoid')(features)

    model = Model(inp, out)
    
    model.summary()
