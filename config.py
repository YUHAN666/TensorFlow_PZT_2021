# coding=utf-8

MODE = {0: "train_segmentation",    # 训练模型分割部分
        1: "train_decision",        # 训练模型分类部分
        2: "visualization",         # 验证模型分割效果
        3: "testing",               # 验证模型分类效果
        4: "savePb",                # 将模型保存为pb文件
        5: "testPb",                # 测试pb模型效果
        6: "view_dataset"           # 使用TensorBoard观察DataSet
        }

DefaultParam = {
    "mode": MODE[0],
    "anew": True,  # 是否需要新建模型，False则新建模型，True则从checkpoint中读取参数
    "name": "p08",
    "balanced_mode": False,  # 是否需要平衡样本

    # Model param
    "momentum": 0.9,                # BatchNorm层动量参数
    "backbone": "ghostnet",         # 骨架网络选择
    "neck": "bifpn",                # 分割头部选择

    # DataManager param
    "image_root_train": "./Data/",        # 训练集图片路径
    "mask_root_train": "./Data/",         # 训练集mask路径
    "image_root_valid": "./Data/",        # 验证集图片路径
    "mask_root_valid": "./Data/",         # 验证集mask路径
    "extension": ".bmp",                  # mask格式
    # "augmentation": ["adjust_gamma", "flip", "rotate", "GaussianBlur", "shift"],  # 图像增强方法选择

    # ImageAugmentor param
    "augmentation_method": {"crop": (10, 10),
                            "rotate": (10, 10),
                            "blur": True,
                            "motion": {"k": (3, 20),
                                       "angle": [0, 90],
                                       "direction": 0}},


    # Trainer param
    "optimizer": "Adam",                # 优化器选择  Adam GD RMS
    "save_frequency": 5,                # 保存checkpoint的频率，训练时每隔多少epoch保存一次模型
    "check_seg_frequency": 50,          # 多少epochs使用TensorBoard检查一次分割效果
    "max_to_keep": 10,                  # 最多保留的checkpoint文件个数
    "epochs": 300,                       # 训练循环数
    "batch_size": 4,                    # 训练batch大小
    "batch_size_inference": 2,          # 保存为pb时batch大小
    "learning_rate": 0.005,              # 学习率
    "lr_decay": "exponential_decay",  # 学习率衰减策略   exponential_decay,inverse_time_decay,natural_exp_decay,cosine_decay
    "stair_case": False,                # 阶梯式衰减
    "decay_rate": 0.7,                  # 衰减率，1则不衰减
    "decay_steps": 10000,               # 衰减步数
    "loss": "cross_entropy",            # 损失函数
    "warm_up": True,  # 预热学习率(先使用较小学习率，warm_up_step后增大至初始学习率以避免nan
    "warm_up_step": 300,  # 预热步数

    # Saver param
    "input_list": ["Image"],        # pb模型输入node
    "output_list": ["decision_out", "mask_out1", "mask_out2"],      # pb模型输出节点
    # "output_list": ["decision_out", "mask_out"],
    "pb_save_path": "./pbMode/",    # pb模型路径
    "pb_name": "123.pb",            # pb模型命名
    "saving_mode": "CBR",           # pb模型存储模式
    "checkpoint_dir": "./checkpoint/",        # checkpoint保存路径

    # pbTester param
    "test_pb_file": "./pbMode/123.pb",      # 需测试的pb模型路径
    "pb_input_tensor": ["Image:0"],         # 需测试的pb模型输入Tensor名
    "pb_output_mask_name": ["mask_out1:0", "mask_out2:0"],  # 需测试的pb模型输出Mask Tensor名
    "pb_output_label_name": ["decision_out:0"],             # 需测试的pb模型输出Label Tensor名
    "timeline_dir": "./timeline/",           # timeline Jason文件输出路径

    # Logger & TensorBoard param
    "log_path": "./Log/",            # Log文件保存路径
    "tensorboard_dir": "./tensorboard/",  # TensorBoard event文件输出路径
    "need_clear_tensorboard": True      # 是否需要清空TensorBoard输出目录下的文件
}

DefaultParam["image_root_train"] += DefaultParam["name"] + "/train_image/"
DefaultParam["image_root_valid"] += DefaultParam["name"] + "/valid_image/"
DefaultParam["mask_root_train"] += DefaultParam["name"] + "/train_mask/"
DefaultParam["mask_root_valid"] += DefaultParam["name"] + "/valid_mask/"
DefaultParam["checkpoint_dir"] += DefaultParam["name"] + "/"

# IMAGE_SIZE的长和宽应该是32的整数倍
if DefaultParam["name"] == "p08":
    IMAGE_SIZE = [576, 320]
elif DefaultParam["name"] == "glue":
    IMAGE_SIZE = [1024, 256]
elif DefaultParam["name"] == "side":
    IMAGE_SIZE = [256, 1280]
elif DefaultParam["name"] == "PZT":
    IMAGE_SIZE = [928, 320]
elif DefaultParam["name"] == "crop":
    IMAGE_SIZE = [256, 1248]
elif DefaultParam["name"] == "p01":
    IMAGE_SIZE = [576, 224]
else:
    raise ValueError("unknown name")

IMAGE_WIDTH = IMAGE_SIZE[1]
IMAGE_HEIGHT = IMAGE_SIZE[0]
BIN_SIZE = [1, 2, 4, 7]
ACTIVATION = 'relu'  # swish mish or relu
ATTENTION = 'se'  # se or cbam
DATA_FORMAT = 'channels_last'
DROP_OUT = False
TRAIN_MODE_IN_TRAIN = True      # 训练时BN层的
TRAIN_MODE_IN_VALID = False
TRAIN_MODE_IN_TEST = False
TEST_RATIO = 0.25
IMAGE_MODE = 'GRAY'  # 0: mono, 1:color
CLASS_NUM = 1
