""" Implementation of Ghostnet """
from collections import namedtuple

import tensorflow as tf

from config import ACTIVATION, ATTENTION
from model.block.GhostNetModule import GhostConv
from model.block.GhostNetModule import GhostDepthConv
from model.block.convblock import ConvBatchNormRelu as CBR
from model.block.moduleCBAM import moduleCBAM
from model.block.squeeze_layers import Squeeze_excitation_layer as SElayer

kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'factor', 'se'])
Bottleneck = namedtuple('Bottleneck', ['kernel', 'stride', 'depth', 'factor', 'se'])

_CONV_DEFS_0 = [
    Conv(kernel=[3, 3], stride=2, depth=16, factor=1, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=16, factor=1, se=0),

    Bottleneck(kernel=[3, 3], stride=2, depth=24, factor=48 / 16, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=24, factor=72 / 24, se=0),

    Bottleneck(kernel=[5, 5], stride=2, depth=40, factor=72 / 24, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=40, factor=120 / 40, se=1),

    Bottleneck(kernel=[3, 3], stride=2, depth=80, factor=240 / 40, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=200 / 80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),
    Bottleneck(kernel=[3, 3], stride=1, depth=80, factor=184 / 80, se=0),

    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=480 / 80, se=1),
    Bottleneck(kernel=[3, 3], stride=1, depth=112, factor=672 / 112, se=1),
    Bottleneck(kernel=[5, 5], stride=2, depth=160, factor=672 / 112, se=1),

    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=0),
    Bottleneck(kernel=[5, 5], stride=1, depth=160, factor=960 / 160, se=1),

]


def ghostnet_base(inputs,
                  mode,
                  data_format,
                  min_depth=8,
                  depth_multiplier=1.0,
                  conv_defs=None,
                  output_stride=None,
                  dw_code=None,
                  ratio_code=None,
                  se=1,
                  scope=None,
                  is_training=False,
                  momentum=0.9):
    """ By adjusting depth_multiplier can change the depth of network """
    if data_format == 'channels_first':
        axis = 1
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
    else:
        axis = -1

    output_layers = []

    def depth(d):
        d = max(int(d * depth_multiplier), min_depth)
        d = round(d / 4) * 4
        return d

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if conv_defs is None:
        conv_defs = _CONV_DEFS_0
    if dw_code is None or len(dw_code) < len(conv_defs):
        dw_code = [3] * len(conv_defs)
    if ratio_code is None or len(ratio_code) < len(conv_defs):
        ratio_code = [2] * len(conv_defs)
    se_code = [x.se for x in conv_defs]
    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, [inputs]):
        net = inputs
        in_depth = 3
        gi = 0

        for i, conv_def in enumerate(conv_defs):
            layer_stride = conv_def.stride
            if layer_stride != 1:
                output_layers.append(net)  # backboneÿ�ν�����֮ǰ�Ĳ���Ϊ���������

            if isinstance(conv_def, Conv):
                net = CBR(net, depth(conv_def.depth), conv_def.kernel[0], strides=conv_def.stride, training=is_training,
                          momentum=momentum, mode=mode, name='ConV_layers{}'.format(i), padding='same',
                          data_format=data_format,
                          activation=ACTIVATION, bn=True, use_bias=False)
            elif isinstance(conv_def, Bottleneck):
                if layer_stride == 1 and in_depth == conv_def.depth:
                    res = None
                else:
                    res = GhostDepthConv(net, conv_def.kernel, stride=layer_stride, data_format=data_format,
                                         name='Bottleneck{}_res_depthwise'.format(i))
                    res = tf.layers.batch_normalization(res, training=is_training,
                                                        name='Bottleneck{}_res_depthwise_BN'.format(i), axis=axis)
                    res = CBR(res, depth(conv_def.depth), 1, 1, training=is_training, momentum=momentum, mode=mode,
                              name='Bottleneck{}_res'.format(i), padding='same',
                              data_format=data_format, activation=None, bn=True, use_bias=False)

                # Increase depth with 1x1 conv.
                net = GhostConv('Bottleneck{}_up_pointwise'.format(i), net, depth(in_depth * conv_def.factor), 1,
                                dw_code[gi], ratio_code[gi], mode=mode, strides=1, data_format=data_format,
                                use_bias=False, is_training=is_training, activation=ACTIVATION, momentum=momentum)

                # DepthWise conv2d.
                if layer_stride > 1:
                    net = GhostDepthConv(net, conv_def.kernel, stride=layer_stride, data_format=data_format,
                                         name='Bottleneck{}_depthwise'.format(i))
                    net = tf.layers.batch_normalization(net, training=is_training,
                                                        name='Bottleneck{}_depthwise_BN'.format(i), axis=axis)

                # SE
                if se_code[i] > 0 and se > 0:
                    if ATTENTION == 'se':
                        # net = SELayer(net, depth(in_depth * conv_def.factor), 4)
                        net = SElayer(net, depth(in_depth * conv_def.factor), depth(in_depth * conv_def.factor) // 4,
                                      "se{}".format(i), data_format=data_format)
                    elif ATTENTION == 'cbma':
                        net = moduleCBAM(net, depth(in_depth * conv_def.factor), depth(in_depth * conv_def.factor) // 4,
                                         'cbma{}'.format(str(i)))

                # Downscale 1x1 conv.
                net = GhostConv('Bottleneck{}_down_pointwise'.format(i), net, depth(conv_def.depth), 1, dw_code[gi],
                                ratio_code[gi], mode=mode, strides=1, data_format=data_format, use_bias=False,
                                is_training=is_training, activation=None, momentum=momentum)
                gi += 1

                # Residual connection
                net = tf.add(res, net, name='Bottleneck{}_Add'.format(i)) if res is not None else net

            in_depth = conv_def.depth
        output_layers.pop(0)  # ��Ҫ��һ�������㣨̫��
        output_layers.append(net)
        return output_layers
