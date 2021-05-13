import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from config import DefaultParam


class ImageAugmentor(object):

    def __init__(self, param):

        self.augmentation_methods = []
        self.aug_method = param["augmentation_method"]

    def transform_seg(self, img, msk):

        image = img.copy()
        mask = msk.copy()
        flag = np.sum(mask)
        self.augmentation_methods.clear()

        segmap = SegmentationMapsOnImage(mask, shape=image.shape)

        if self.aug_method["crop"] and np.random.uniform() > 0.5:
            self.augmentation_methods.append(iaa.Crop(px=self.aug_method["crop"]))

        if self.aug_method["rotate"] and np.random.uniform() > 0.5:
            self.augmentation_methods.append(iaa.Affine(rotate=self.aug_method["rotate"]))

        if self.aug_method["blur"] and np.random.uniform() > 0.5:
            self.augmentation_methods.append(iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                                                        iaa.AverageBlur(k=(2, 7))]))

        if self.aug_method["motion"] and np.random.uniform() > 0.5:
            self.augmentation_methods.append(iaa.MotionBlur(k=self.aug_method["motion"]["k"],
                                                            angle=self.aug_method["motion"]["angle"],
                                                            direction=self.aug_method["motion"]["direction"]))

        augmentation = iaa.Sequential(self.augmentation_methods, random_order=True)
        image, mask = augmentation(image=image, segmentation_maps=segmap)
        mask = mask.draw(size=image.shape[:2])
        mask = np.where(mask[0] > 0, 1, 0)
        mask = np.array(mask[:, :, 0], dtype=np.float64)
        auged_flag = np.sum(mask)

        if flag != 0 and auged_flag == 0:  # 确认增强后目标位置是否偏出图像范围
            msk = np.where(msk > 0, 1, 0)
            return img, msk

        return image, mask


if __name__ == "__main__":
    image_augmentor = ImageAugmentor(DefaultParam)
    o = image_augmentor.transform_seg()
