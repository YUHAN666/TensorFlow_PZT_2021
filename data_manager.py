import os

import cv2
import numpy as np
import tensorflow as tf

from config import *
from image_augmentor import ImageAugmentor


class DataManager(object):
    def __init__(self, param, shuffle=True, valid=False, extension='.bmp'):

        self.shuffle = shuffle
        self.extension = extension
        if valid:
            self.image_root = param["image_root_valid"]
            self.mask_root = param["mask_root_valid"]
            # self.batch_size = 1
        else:
            self.image_root = param["image_root_train"]
            self.mask_root = param["mask_root_train"]

        self.mode = param["mode"]
        if self.mode == "train_segmentation" or self.mode == "train_decision":
            self.batch_size = param["batch_size"]
        elif self.mode == "savePb" or self.mode == "testPb":
            self.batch_size = param["batch_size_inference"]
        else:
            self.batch_size = 1
        self.epoch_num = param["epochs"]
        self.augmentor = ImageAugmentor(param)
        # self.augmentation = param["augmentation"]
        self.next_batch = self.get_next()
        self.image_files = [x[2] for x in os.walk(self.image_root)][0]
        self.mask_files = [x[2] for x in os.walk(self.mask_root)][0]
        if param["balanced_mode"]:
            self.image_files = [x for x in self.image_files if "p_" in x]
            self.mask_files = [x for x in self.mask_files if "p_" in x]

        self.num_batch = len(self.image_files)//self.batch_size

    def get_next(self):
        """ Encapsulate generator into TensorFlow DataSet"""
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32, tf.string))
        dataset = dataset.repeat(self.epoch_num+self.epoch_num//10+1)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    def generator(self):
        """
        Generator of image, mask, label, and image_root
        Should be revised according to the saving path of data
        :return:  image
                  mask
                  label
                  image_path
        """
        rand_index = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(rand_index)
        for index in rand_index:
            image_filename = self.image_files[index]
            # 训练数据
            image_path = self.image_root + image_filename
            if self.check_mask_name():
                mask_path = self.mask_root + image_filename
            else:
                mask_path = self.mask_root + image_filename.split(".")[0] + "_label" + self.extension

            " Generate label from image filename. Shall be revised according to specific situation"
            if image_path.split('/')[-1].split('_')[0] == 'n':
                label = np.array([0.0])
            else:
                label = np.array([1.0])

            image, mask = self.read_data(image_path, mask_path)

            image = image / 255.0
            mask = mask // 255

            if self.mode == "train_segmentation" or self.mode == "train_decision":
                aug_random = np.random.uniform()
                if aug_random > 0.9:
                    image, mask = self.augmentor.transform_seg(image, mask)
                    # # adjust_gamma
                    # if np.random.uniform() > 0.7 and "adjust_gamma" in self.augmentation:
                    #     expo = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                    #     image = exposure.adjust_gamma(image, expo)
                    #
                    # # flip
                    # if np.random.uniform() > 0.7 and "flip" in self.augmentation:
                    #     aug_seed = np.random.randint(-1, 2)
                    #     image = cv2.flip(image, aug_seed)
                    #     mask = cv2.flip(mask, aug_seed)
                    #
                    # # rotate
                    # if np.random.uniform() > 0.7 and "rotate" in self.augmentation:
                    #     angle = np.random.randint(-5, 5)
                    #     image = self.rotate(image, angle)
                    #     mask = self.rotate(mask, angle)
                    #
                    # # GassianBlur
                    # if np.random.uniform() > 0.7 and "GaussianBlur" in self.augmentation:
                    #     image = cv2.GaussianBlur(image, (5, 5), 0)
                    #
                    # # shift
                    # if np.random.uniform() > 0.7 and "shift" in self.augmentation:
                    #     dx = np.random.randint(-5, 5)  # width*5%
                    #     dy = np.random.randint(-5, 5)  # Height*10%
                    #     rows, cols = image.shape[:2]
                    #     M = np.float32([[1, 0, dx], [0, 1, dy]])  # (x,y) -> (dx,dy)
                    #     image = cv2.warpAffine(image, M, (cols, rows))
                    #     mask = cv2.warpAffine(mask, M, (cols, rows))

            if len(image.shape) == 2:
                image = (np.array(image[:, :, np.newaxis]))
            if len(mask.shape) == 2:
                mask = (np.array(mask[:, :, np.newaxis]))

            yield image, mask, label, image_path

    @staticmethod
    def read_data(image_path, mask_path):
        """ Read image and mask"""
        img = cv2.imread(image_path, 0)  # /255.#read the gray image
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        try:
            msk = cv2.imread(mask_path, 0)  # /255.#read the gray image
            msk = cv2.resize(msk, (IMAGE_WIDTH, IMAGE_HEIGHT))
            _, msk = cv2.threshold(msk, 0, 255, cv2.THRESH_BINARY)
        except:
            raise ValueError(" Cannot find mask {}".format(mask_path))

        return img, msk

    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        """Rotate and scale image around given center at given scale"""
        (h, w) = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def check_mask_name(self):
        if "label" in self.mask_files[0]:
            return False
        else:
            return True
