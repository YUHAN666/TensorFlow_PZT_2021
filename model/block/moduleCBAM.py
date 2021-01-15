"""
Implementation of CBAM module
"""
import tensorflow as tf

from config import ACTIVATION
from model.activation.mish import mish
from model.activation.swish import swish


def channelAttention(input_tensor, out_dim, se_dim):
    with tf.variable_scope('channel_attention'):
        x1 = tf.reduce_mean(input_tensor, axis=[1, 2], keep_dims=True)
        x2 = tf.reduce_max(input_tensor, axis=[1, 2], keep_dims=True)
        # out_dim = input_tensor.shape[-1]

        x1 = tf.layers.conv2d(x1, se_dim, (1, 1), use_bias=False, name='dense1', reuse=None)
        if ACTIVATION == 'swish':
            x1 = swish(x1, 'swish')
        elif ACTIVATION == 'mish':
            x1 = mish(x1, 'mish')
        else:
            x1 = tf.nn.relu(x1)
        x1 = tf.layers.conv2d(x1, out_dim, (1, 1), use_bias=False, name='dense2', reuse=None)

        x2 = tf.layers.conv2d(x2, se_dim, (1, 1), use_bias=False, name='dense1', reuse=True)
        if ACTIVATION == 'swish':
            x2 = swish(x2, 'swish')
        elif ACTIVATION == 'mish':
            x2 = mish(x2, 'mish')
        else:
            x2 = tf.nn.relu(x2)
        x2 = tf.layers.conv2d(x2, out_dim, (1, 1), use_bias=False, name='dense2', reuse=True)

        x = x1 + x2
        x = tf.nn.sigmoid(x)

        x = input_tensor * x
    return x


def spatialAttention(input_tensor):
    with tf.variable_scope('spatial_attention'):
        x1 = tf.reduce_max(input_tensor, axis=-1, keep_dims=True)
        x2 = tf.reduce_mean(input_tensor, axis=-1, keep_dims=True)

        x = tf.concat([x1, x2], axis=-1)
        x = tf.layers.conv2d(x, 1, (1, 1), use_bias=False)
        x = tf.nn.sigmoid(x)

        x = input_tensor * x

    return x


def moduleCBAM(input_tensor, out_dim, se_dim, name):
    with tf.variable_scope(name):
        x = channelAttention(input_tensor, out_dim, se_dim)
        x = spatialAttention(x)

    return x


def modified_SAM(input_tensor):
    channel = input_tensor.shape[-1]
    x = tf.layers.conv2d(input_tensor, channel, (1, 1))
    x = tf.nn.sigmoid(x)

    return tf.multiply(x, input_tensor)
