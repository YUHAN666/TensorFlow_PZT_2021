from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import cv2
from config import *
from timeit import default_timer as timer
from utiles.iouEval import iouEval
from tqdm import tqdm
from utiles.utils import concatImage
import time
from data_manager import DataManager
from tensorflow.python.client import timeline
import numpy as np


class PbTester(object):

    def __init__(self, param,  logger, pb_test_dir="./pb_test_result/"):
        tf.reset_default_graph()
        self.pb_file_path = param["test_pb_file"]
        if not os.path.exists(pb_test_dir):
            os.makedirs(pb_test_dir)
        self.pb_test_dir = pb_test_dir
        self.timeline_dir = param["timeline_dir"]
        self.input_tensor_name = param["pb_input_tensor"]
        self.output_mask_name = param["pb_output_mask_name"]
        self.output_label_name = param["pb_output_label_name"]
        self.data_manager = DataManager(param, shuffle=False)
        self.session = tf.Session()
        self.logger = logger
        print(" Loading model from {}".format(self.pb_file_path))
        with gfile.FastGFile(self.pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.session.graph.as_default()
            tf.import_graph_def(graph_def, name='')

    def test_segmentation(self):

        print(" Testing segmentation ability of pb model, pictures are generated to {}".format(self.pb_test_dir))
        with self.session.as_default():
            input_image = self.session.graph.get_tensor_by_name(self.input_tensor_name[0])
            mask_out = []
            for m in self.output_mask_name:
                mask_out.append(self.session.graph.get_tensor_by_name(m))
            num_step = 0.0
            total_time = 0.0
            iouGen = iouEval(1)
            time.sleep(0.1)
            pbar = tqdm(total=self.data_manager.num_batch * self.data_manager.batch_size, leave=True)
            for batch in range(self.data_manager.num_batch):
                img_batch, mask_batch, label_batch, path = self.session.run(self.data_manager.next_batch)
                start = timer()
                masks = self.session.run(mask_out, feed_dict={input_image: img_batch})
                end = timer()
                t = end - start
                for i in range(len(masks)):
                    filename = str(path[i]).split('/')[-1].split('\\')[0].split("'")[0]
                    if IMAGE_MODE == 0:
                        image = np.array(img_batch[i]).squeeze()
                    else:
                        image = np.mean(img_batch[i], axis=2)
                    mask = np.array(masks[i]).squeeze(2) * 255
                    mask_in = np.array(mask_batch[i]).squeeze(2) * 255
                    if image.shape[0] < image.shape[1]:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        mask_in = cv2.rotate(mask_in, cv2.ROTATE_90_CLOCKWISE)

                    image = image * 255
                    # label_pixel = np.array(label_pixel_batch[i]).squeeze(2)*255
                    img_visual = concatImage([image, mask_in, mask])
                    visualization_path = os.path.join(self.pb_test_dir, filename)
                    img_visual.save(visualization_path)
                    iouGen.addBatch(mask_in/255, mask/255)

                total_time += t
                num_step += 1
                pbar.update(self.data_manager.batch_size)
            pbar.clear()
            pbar.close()
            iou = iouGen.getIoU()
            time.sleep(0.1)
            print(" Average IoU:{}, Average time per mask:{}".format(iou, total_time/num_step/len(mask_out)))

    def test_decision(self):
        print(" Testing decision ability of pb model")
        self.logger.info("Testing decision ability of pb model")
        with self.session.as_default():
            input_image = self.session.graph.get_tensor_by_name(self.input_tensor_name[0])
            decision_out = self.session.graph.get_tensor_by_name(self.output_label_name[0])
            num_step = 0.0
            total_time = 0.0
            true_account = 0
            false_account = 0
            false_path = []
            false_decision = []
            time.sleep(0.1)
            pbar = tqdm(total=self.data_manager.num_batch * self.data_manager.batch_size, leave=True)
            for batch in range(self.data_manager.num_batch):
                img_batch, mask_batch, label_batch, path = self.session.run(self.data_manager.next_batch)
                start = timer()
                decision = self.session.run(decision_out, feed_dict={input_image: img_batch})
                end = timer()
                t = end - start
                for i in range(self.data_manager.batch_size):
                    if (decision[i] > 0.5 and label_batch[i] == 1) or (decision[i] <= 0.5 and label_batch[i] == 0):
                        true_account += 1
                    else:
                        false_account += 1
                        false_path.append(path[i])
                        false_decision.append(decision[i])
                total_time += t
                num_step += 1
                pbar.update(self.data_manager.batch_size)
            accuracy = true_account/(true_account + false_account)
            pbar.clear()
            pbar.close()
            time.sleep(0.1)
            print("Accuracy:{}, Average time per decision:{}".format(accuracy, total_time / num_step / self.data_manager.batch_size))
            self.logger.info("Accuracy:{}, Average time per decision:{}".format(accuracy,total_time / num_step / self.data_manager.batch_size))
            self.logger.info("False account: {}, path: {}, decision_out: {}".format(false_account, false_path, false_decision))

    def view_timeline(self):
        """View timeline of pb model"""
        # 使用chrome浏览器输入chrome://tracing/
        if not os.path.exists(self.timeline_dir):
            os.makedirs(self.timeline_dir)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        self.session.run(tf.global_variables_initializer())
        input_image_tensor = self.session.graph.get_tensor_by_name(self.input_tensor_name[0])
        # �����������������
        output_class = self.session.graph.get_tensor_by_name(self.output_label_name[0])
        # ��ȡ����ͼƬ
        for i in range(self.data_manager.num_batch):

            img_batch, mask_batch, label_batch, path = self.session.run(self.data_manager.next_batch)
            start = time.process_time()
            whichclass = self.session.run([output_class], feed_dict={input_image_tensor: img_batch},
                                          options=options, run_metadata=run_metadata)
            elapsed = (time.process_time() - start)
            print("time used:", elapsed)
            print("out:{}".format(whichclass))

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('./timeline/timeline_02_step_{}.json'.format(i), 'w') as f:
                f.write(chrome_trace)

