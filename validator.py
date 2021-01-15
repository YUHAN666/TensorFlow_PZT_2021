import os
import numpy as np
import cv2
from config import IMAGE_MODE
from utiles.utils import concatImage
from timeit import default_timer as timer
from tqdm import tqdm
import time


class Validator(object):
    def __init__(self, sess, model, logger):
        """ Validate the performance of model from checkpoint when training process is complete"""
        self.session = sess
        self.model = model
        self.logger = logger

    def valid_segmentation(self, data_manager, save_dir="./visualization"):
        """ Validate the segmentation performance of model by generating pictures made up by: image_in, mask_in, and mask_out"""
        print(" Start generating mask with original input to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time.sleep(0.1)
        pbar = tqdm(total=data_manager.num_batch * data_manager.batch_size, leave=True)
        for batch in range(data_manager.num_batch):

            img_batch, mask_in_batch, _, path = self.session.run(data_manager.next_batch)
            mask_out = self.session.run(self.model.mask_out, feed_dict={self.model.image_input: img_batch})

            for i in range(data_manager.batch_size):
                filename = str(path[i]).split('/')[-1].split('\\')[0].split("'")[0]
                if IMAGE_MODE == 0:
                    image = np.array(img_batch[i]).squeeze()
                else:
                    image = np.mean(img_batch[i], axis=2)
                mask = np.array(mask_out[i]).squeeze(2) * 255
                mask_in = np.array(mask_in_batch[i]).squeeze(2) * 255
                if image.shape[0] < image.shape[1]:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                    mask_in = cv2.rotate(mask_in, cv2.ROTATE_90_CLOCKWISE)

                image = image * 255
                # label_pixel = np.array(label_pixel_batch[i]).squeeze(2)*255
                img_visual = concatImage([image, mask_in, mask])

                # size = Image.fromarray(mask).size
                # img_visual = Image.fromarray(mask).resize(size, Image.BILINEAR)
                # target = Image.new("L", (size[0], size[1] * 1))
                # target.paste(img_visual, (0 * size[0], 0, (0 + 1) * size[0], size[1]))

                visualization_path = os.path.join(save_dir, filename)
                img_visual.save(visualization_path)
            pbar.update(data_manager.batch_size)
        pbar.close()

    def valid_decision(self, data_manager):
        """ Validate the classification performance of model"""
        print(" Start testing classification ability of model")
        self.logger.info("Start testing classification ability of model ")
        true = 0
        false = 0
        total_time = 0
        time.sleep(0.1)
        pbar = tqdm(total=data_manager.num_batch * data_manager.batch_size, leave=True)
        false_list = []
        false_decision = []
        for batch in range(data_manager.num_batch):

            img_batch, mask_in_batch, label_batch, path = self.session.run(data_manager.next_batch)
            start = timer()
            decision_out = self.session.run(self.model.decision_out, feed_dict={self.model.image_input: img_batch})
            end = timer()
            t = end - start
            total_time += t
            for i in range(data_manager.batch_size):
                if (label_batch[i] == 1 and decision_out[i] > 0.5) or (label_batch[i] == 0 and decision_out[i] <= 0.5):
                    true += 1
                else:
                    false_list.append(path[i])
                    false_decision.append(decision_out[i])
                    false += 1
            pbar.update(data_manager.batch_size)
        pbar.close()
        time.sleep(0.1)
        print("accuracy:{}, time per batch:{}, time per image:{}".format(true/(true+false),
                                                                         total_time/data_manager.num_batch,
                                                                         total_time/data_manager.num_batch/data_manager.batch_size))
        print("false number:{}, path:{}, decision_out:{}".format(false, false_list, false_decision))
        self.logger.info("accuracy:{}, time per batch:{}, time per image:{}".format(true/(true+false),
                                                                         total_time/data_manager.num_batch,
                                                                         total_time/data_manager.num_batch/data_manager.batch_size))
        self.logger.info("false number:{}, path:{}, decision_out:{}".format(false, false_list, false_decision))
