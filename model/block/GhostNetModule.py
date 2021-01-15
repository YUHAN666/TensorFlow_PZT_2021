""" Implementation of modules use in GhostNet.py """
import math
import tensorflow as tf

from model.activation.mish import mish
from model.activation.swish import swish
from model.block.convblock import ConvBatchNormRelu as CBR

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)


def GhostDepthConv(x, kernel_shape, channel_mult=1, padding='SAME', stride=1, rate=1, data_format='channels_first',
                   W_init=None, activation=tf.identity, name=None):
    in_shape = x.get_shape().as_list()

    if data_format == 'channels_first':
        in_channel = in_shape[1]
        stride_shape = [1, 1, stride, stride]
        data_format = 'NCHW'
    else:
        in_channel = in_shape[-1]
        stride_shape = [1, stride, stride, 1]
        data_format = 'NHWC'

    out_channel = in_channel * channel_mult

    if W_init is None:
        W_init = kernel_initializer
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable(name + "_weight", filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, stride_shape, padding=padding, rate=[rate, rate], data_format=data_format,
                                  name=name + "_depthwise_conv2d")

    return conv


def GhostConv(name, x, filters, kernel_size, dw_size, ratio, mode, padding='SAME', strides=1,
              data_format='channels_first', use_bias=False, is_training=False, activation='relu', momentum=0.9):
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = -1

    with tf.variable_scope(name):
        init_channels = math.ceil(filters / ratio)

        x = CBR(x, init_channels, kernel_size, strides=strides, training=is_training, momentum=momentum, mode=mode,
                name=name, padding='same', data_format=data_format, activation='relu', bn=True, use_bias=use_bias)

        if ratio == 1:
            return x
        dw1 = GhostDepthConv(x, [dw_size, dw_size], channel_mult=ratio - 1, stride=1, data_format=data_format,
                             name=name)
        dw1 = tf.layers.batch_normalization(dw1, training=is_training, name=name + 'BN_2', axis=axis)
        if activation == 'relu':
            dw1 = tf.nn.relu(dw1, name=name + 'relu_2')
        elif activation == 'mish':
            dw1 = mish(dw1, name='mish_2')
        elif activation == 'swish':
            dw1 = swish(dw1, name=name + 'swish_2')
        else:
            pass

        if data_format == 'channels_first':
            dw1 = dw1[:, :filters - init_channels, :, :]
        else:
            dw1 = dw1[:, :, :, :filters - init_channels]
        x = tf.concat([x, dw1], axis=axis)
        return x
