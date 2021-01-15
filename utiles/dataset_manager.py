import os
import shutil

import numpy as np


class DataSetManager(object):

    def __init__(self):
        pass

    def split_data(self, image_root, mask_root, dataset_root, valid_ratio, test_ratio):

        if not self.match_mask(image_root, mask_root):

            image_files = [x[2] for x in os.walk(image_root)][0]
            mask_files = [x[2] for x in os.walk(mask_root)][0]
            image_num = len(image_files)
            valid_num = int(image_num * valid_ratio)
            test_num = int(image_num * test_ratio)
            train_num = image_num - valid_num - test_num

            np.random.shuffle(image_files)

            train_files = image_files[:train_num]
            valid_files = image_files[train_num:train_num + valid_num]
            test_files = image_files[train_num + valid_num:]

            train_image_dir = dataset_root + "/train_image/"
            train_mask_dir = dataset_root + "/train_mask/"
            valid_image_dir = dataset_root + "/valid_image/"
            valid_mask_dir = dataset_root + "/valid_mask/"
            test_image_dir = dataset_root + "/test_image/"
            test_mask_dir = dataset_root + "/test_mask/"

            dirs = [train_image_dir, train_mask_dir, valid_image_dir, valid_mask_dir, test_image_dir, test_mask_dir]

            for d in dirs:
                if not os.path.exists(d):
                    os.makedirs(d)

            for path in train_files:
                image_path = image_root + path
                if self.check_mask_name(mask_files[0]):
                    mask_path = mask_root + path
                else:
                    mask_path = mask_root + path.split(".")[0] + "_label.bmp"

                shutil.move(image_path, train_image_dir)
                shutil.move(mask_path, train_mask_dir)

            for path in valid_files:
                image_path = image_root + path
                if self.check_mask_name(mask_files[0]):
                    mask_path = mask_root + path
                else:
                    mask_path = mask_root + path.split(".")[0] + "_label.bmp"

                shutil.move(image_path, valid_image_dir)
                shutil.move(mask_path, valid_mask_dir)

            for path in test_files:
                image_path = image_root + path
                if self.check_mask_name(mask_files[0]):
                    mask_path = mask_root + path
                else:
                    mask_path = mask_root + path.split(".")[0] + "_label.bmp"

                shutil.move(image_path, test_image_dir)
                shutil.move(mask_path, test_mask_dir)

        else:
            print("Missing masks:{}".format(self.match_mask(image_root, mask_root)))

    @staticmethod
    def match_mask(image_root, mask_root):

        image_files = [x[2] for x in os.walk(image_root)][0]
        image_files = [i.split(".")[0] for i in image_files]
        mask_files = [x[2] for x in os.walk(mask_root)][0]
        mask_files = [i.split(".")[0] for i in mask_files]
        miss_masks = []

        for i in image_files:
            if i not in mask_files and (i + "_label") not in mask_files:
                miss_masks.append(i)

        return miss_masks

    @staticmethod
    def check_mask_name(path):
        if "label" in path:
            return False
        else:
            return True


if __name__ == "__main__":
    dataset_manager = DataSetManager()
    image_root = "E:/CODES/TensorFlow_PZT/Data/p01/image/"
    mask_root = "E:/CODES/TensorFlow_PZT/Data/p01/mask/"
    dataset_root = "E:/CODES/TensorFlow_PZT/Data/p01/"
    dataset_manager.split_data(image_root, mask_root, dataset_root, 0.2, 0)
