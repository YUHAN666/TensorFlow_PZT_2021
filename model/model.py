import tensorflow as tf

from config import CLASS_NUM, IMAGE_SIZE, IMAGE_MODE, DATA_FORMAT, ACTIVATION
from model.dec_head.decision_head import decision_head


class Model(object):

    def __init__(self, sess, param, logger):
        self.step = 0
        self.session = sess
        self.bn_momentum = param["momentum"]
        self.mode = param["mode"]
        self.backbone = param["backbone"]
        self.neck = param["neck"]
        self.logger = logger
        self.num_channel = 1 if IMAGE_MODE == 'GRAY' else 3
        self.checkPoint_dir = param["checkpoint_dir"]
        self.logger.info("Building model... backbone:{}, neck:{}".format(self.backbone, self.neck))
        if param["mode"] == "train_segmentation":
            self.keep_dropout_backbone = True
            self.keep_dropout_head = True
        elif param["mode"] == "train_decision":
            self.keep_dropout_backbone = False
            self.keep_dropout_head = True
        else:
            self.keep_dropout_backbone = False
            self.keep_dropout_head = False
        self.batch_size = param["batch_size"]
        self.batch_size_inference = param["batch_size_inference"]

        with self.session.as_default():
            # Build placeholder to receive data
            if self.mode == 'train_segmentation' or self.mode == 'train_decision':
                self.is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
                self.is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')

                self.image_input = tf.placeholder(tf.float32, shape=(self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1],
                                                                     self.num_channel), name='Image')

                self.label = tf.placeholder(tf.float32, shape=(self.batch_size, CLASS_NUM), name='Label')
                self.mask = tf.placeholder(tf.float32, shape=(self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CLASS_NUM),
                                           name='mask')

            elif self.mode == "savePb":
                self.is_training_seg = False
                self.is_training_dec = False

                self.image_input = tf.placeholder(tf.float32,
                                                  shape=(self.batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1],
                                                         self.num_channel), name='Image')
            else:
                self.is_training_seg = False
                self.is_training_dec = False

                self.image_input = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], self.num_channel),
                                                  name='Image')
                self.mask = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], self.num_channel),
                                           name='mask')

            # Building model graph
            self.segmentation_output, self.decision_output, self.mask_out, self.decision_out = self.build_model()

    def build_model(self):
        """
        Build model graph in session
        You can choose different backbone by setting "backbone" in param,
        which includes mixnet(unofficial version), mixnet_official(the official version), efficientnet, fast_scnn, sinet,
        ghostnet, sinet, lednet, cspnet
        :return: segmentation_output: nodes for calculating segmentation loss
                 decision_output: nodes for calculating decision loss
                 mask_out: nodes for visualization output mask of the model
        """
        # create backbone
        if self.backbone == 'mixnet':
            from model.backbone.MixNet import MixNetSmall
            backbone_output = MixNetSmall(self.image_input, scope='mixnet_backbone', include_top=False,
                                          keep_dropout_backbone=self.keep_dropout_backbone,
                                          training=self.is_training_seg)

        elif self.backbone == 'mixnet_official':
            from model.backbone.MixNet_official import build_model_base
            backbone_output = build_model_base(self.image_input, 'mixnet-s', training=self.is_training_seg,
                                               override_params=None, scope='mixnet_backbone')

        elif self.backbone == 'efficientnet':
            from model.backbone.efficientnet import EfficientNetB0
            backbone_output = EfficientNetB0(self.image_input, model_name='efficientnet_backbone',
                                             input_shape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], self.num_channel),
                                             keep_dropout_backbone=self.keep_dropout_backbone,
                                             training=self.is_training_seg, include_top=True, classes=CLASS_NUM)

        elif self.backbone == 'fast_scnn':
            from model.backbone.fast_scnn import build_fast_scnn
            backbone_output = build_fast_scnn(self.image_input, 'fast_scnn_backbone', is_training=self.is_training_seg,
                                              keep_dropout_backbone=self.keep_dropout_backbone)

        elif self.backbone == 'ghostnet':
            from model.backbone.GhostNet import ghostnet_base
            # Set depth_multiplier to change the depth of GhostNet
            backbone_output = ghostnet_base(self.image_input, mode=self.mode, data_format=DATA_FORMAT,
                                            scope='ghostnet_backbone',
                                            dw_code=None, ratio_code=None,
                                            se=1, min_depth=8, depth_multiplier=0.5, conv_defs=None,
                                            is_training=self.is_training_seg, momentum=self.bn_momentum)

        elif self.backbone == 'sinet':
            from model.backbone.SiNet import SINet
            backbone_output = SINet(self.image_input, classes=CLASS_NUM, p=2, q=8, chnn=1, training=True,
                                    bn_momentum=0.99, scope='sinet_backbone', reuse=False)

        elif self.backbone == 'lednet':
            from model.backbone.LEDNet import lednet
            backbone_output = lednet(self.image_input, training=self.is_training_seg, scope='lednet_backbone',
                                     keep_dropout=False)

        elif self.backbone == 'cspnet':
            from model.backbone.CSPDenseNet import CSPPeleeNet
            backbone_output = CSPPeleeNet(self.image_input, data_format=DATA_FORMAT, drop_rate=0.0,
                                          training=self.is_training_seg, momentum=self.bn_momentum,
                                          name="cspnet_backbone", mode=self.mode, activation=ACTIVATION)

        else:
            raise ValueError("Unknown Backbone")

        segmentation_output, decision_in = self.build_segmentation_head(backbone_output, self.is_training_seg)

        decision_output = self.build_decision_head(decision_in)  # ���ڼ���loss�����decision_loss�����¿������䵼��
        decision_out = tf.nn.sigmoid(decision_output, name='decision_out')

        if DATA_FORMAT == 'channels_first':
            logits_pixel = tf.transpose(segmentation_output[0], [0, 2, 3, 1])
        else:
            logits_pixel = segmentation_output[0]
        logits_pixel = tf.image.resize_images(logits_pixel, (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True,
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_out = tf.nn.sigmoid(logits_pixel, name='mask_out')
        if self.mode == "savePb":
            mask_out = [tf.nn.sigmoid(logits_pixel[0], name='mask_out1'),
                        tf.nn.sigmoid(logits_pixel[1], name='mask_out2')]

        return segmentation_output, decision_output, mask_out, decision_out

    def build_segmentation_head(self, backbone_output, is_training_seg):
        """
        Build segmentation head
        You can choose different segmentation head by setting "neck" in param, which includes bifpn, fpn, bfp, pan
        :return: segmentation_out: list of output nodes for calculate loss
                 decision_in: list of output nodes for decision head
        """
        with tf.variable_scope('segmentation_head'):
            if len(backbone_output) == 5:
                if self.neck == 'bifpn':
                    from model.seg_head.FPN import bifpn_neck
                    P3_out, P4_out, P5_out, P6_out, P7_out = bifpn_neck(backbone_output, 64,
                                                                        is_training=is_training_seg,
                                                                        momentum=self.bn_momentum,
                                                                        mode=self.mode, data_format=DATA_FORMAT)
                    P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3',
                                          data_format=DATA_FORMAT)
                    P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4',
                                          data_format=DATA_FORMAT)
                    P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5',
                                          data_format=DATA_FORMAT)
                    P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6',
                                          data_format=DATA_FORMAT)
                    P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7',
                                          data_format=DATA_FORMAT)
                    segmentation_out = [P3, P4, P5, P6, P7]
                    decision_in = [P3_out, P3]
                elif self.neck == 'fpn':
                    from model.seg_head.FPN import fpn_neck

                    segmentation_out = fpn_neck(backbone_output, CLASS_NUM, drop_rate=0.2,
                                                keep_dropout_head=self.keep_dropout_head,
                                                training=is_training_seg)
                    decision_in = segmentation_out
                elif self.neck == 'bfp':
                    from model.seg_head.BFP import bfp_segmentation_head

                    P3_out, P4_out, P5_out, P6_out, P7_out = bfp_segmentation_head(backbone_output, 64,
                                                                                   is_training=is_training_seg)
                    P3 = tf.layers.conv2d(P3_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P3')
                    P4 = tf.layers.conv2d(P4_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P4')
                    P5 = tf.layers.conv2d(P5_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P5')
                    P6 = tf.layers.conv2d(P6_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P6')
                    P7 = tf.layers.conv2d(P7_out, CLASS_NUM, (1, 1), (1, 1), use_bias=False, name='P7')
                    segmentation_out = [P3, P4, P5, P6, P7]
                    decision_in = [P3_out, P3]
                elif self.neck == 'pan':
                    from model.seg_head.FPN import PAN
                    backbone_output = PAN(backbone_output, 128, training=self.is_training_seg)

                    seg_fea = backbone_output[-1]
                    seg_fea = tf.keras.layers.UpSampling2D((4, 4))(seg_fea)
                    seg_fea = tf.keras.layers.DepthwiseConv2D((3, 3), (1, 1), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)
                    seg_fea = tf.layers.conv2d(seg_fea, 128, (1, 1))

                    seg_fea = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)

                    seg_fea = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same')(seg_fea)
                    seg_fea = tf.layers.batch_normalization(seg_fea, training=is_training_seg)
                    seg_fea = tf.nn.relu(seg_fea)

                    seg_fea1 = tf.layers.conv2d(seg_fea, 1, (1, 1))

                    segmentation_out = [seg_fea, seg_fea1]
                    decision_in = segmentation_out
                else:
                    raise ValueError(" Unknown neck ")

            return segmentation_out, decision_in

    def build_decision_head(self, decision_in, ):
        """
        Build decision head
        :return: output node of decision head
        """
        dec_out = decision_head(decision_in[0], decision_in[1], class_num=CLASS_NUM, scope='decision_head',
                                keep_dropout_head=self.keep_dropout_head,
                                training=self.is_training_dec, data_format=DATA_FORMAT, momentum=self.bn_momentum,
                                mode=self.mode, activation=ACTIVATION)

        return dec_out
