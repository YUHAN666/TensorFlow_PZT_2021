import tensorflow as tf
from data_manager import DataManager
from model.model import Model
from trainer import Trainer
from saver import Saver
from validator import Validator
from pb_tester import PbTester
from tensorboard_manager import TensorboardManager


class Agent(object):

    def __init__(self, param, logger):

        self.logger = logger

        logger.info("Start initializing Agent, mode is {}".format(param["mode"]))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.param = param
        self.session = tf.Session(config=config)
        self.model = Model(self.session, self.param, self.logger)    # 建立将模型graph并写入session中
        self.data_manager_train = DataManager(self.param, shuffle=True)     # 训练集数据生成器
        self.data_manager_valid = DataManager(self.param, shuffle=False, valid=True)    # 验证集数据生成器
        self.tensorboard_manager = TensorboardManager(self.session, self.param, self.model, self.logger)  # 使用TensorBoard进行可视化
        self.trainer = Trainer(self.session, self.model, self.param, self.logger, self.tensorboard_manager)    # 损失函数，优化器，以及训练策略
        self.saver = Saver(self.session, self.param, self.model.checkPoint_dir, self.logger)     # 用于将session保存到checkpoint或pb文件中
        self.validator = Validator(self.session, self.model, self.logger)        # 用于验证训练后模型的性能

        logger.info("Successfully initialized")

    def run(self):

        if not self.param["anew"] and self.param["mode"] != "testPb":
            self.saver.load_checkpoint()

        if self.param["mode"] == "train_segmentation":      # 训练模型分割部分
            self.trainer.train_segmentation(self.data_manager_train, self.data_manager_valid, self.saver)
        elif self.param["mode"] == "train_decision":        # 训练模型分类部分
            self.trainer.train_decision(self.data_manager_train, self.data_manager_valid, self.saver)
        elif self.param["mode"] == "visualization":         # 验证模型分割效果
            self.validator.valid_segmentation(self.data_manager_train)
        elif self.param["mode"] == "testing":               # 验证模型分类效果
            self.validator.valid_decision(self.data_manager_train)
            self.validator.valid_decision(self.data_manager_valid)
        elif self.param["mode"] == "savePb":                # 保存模型到pb文件
            self.saver.save_pb()
        elif self.param["mode"] == "testPb":                # 测试pb模型效果
            self.pb_tester = PbTester(self.param, self.logger)
            self.pb_tester.test_segmentation()
            self.pb_tester.test_decision()
            # self.pb_tester.view_timeline()
        elif self.param["mode"] == "view_dataset":
            # tensorboard --logdir=E:/CODES/TensorFlow_PZT/tensorboard --samples_per_plugin=images=1000
            self.tensorboard_manager.vis_dataset(self.data_manager_train)
            self.tensorboard_manager.vis_mask_out(self.data_manager_train)

        self.session.close()
        self.tensorboard_manager.close()






