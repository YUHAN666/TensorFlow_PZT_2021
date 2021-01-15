import tensorflow as tf
import os
from config import IMAGE_MODE, IMAGE_SIZE


class TensorboardManager(object):

    def __init__(self, sess, param, model, logger):
        self.session = sess
        self.model = model
        self.logger = logger
        self.tensorboard_dir = param["tensorboard_dir"]
        need_clear = param["need_clear_tensorboard"]
        if need_clear:
            for _, _, files in os.walk(self.tensorboard_dir):
                for file in files:
                    os.remove(os.path.join(self.tensorboard_dir, file))
        # tensorboard --logdir=E:\CODES\TensorFlow_PZT\tensorboard
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.session.graph)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        self.num_channel = 1 if IMAGE_MODE == 'GRAY' else 3
        self.axis = 1 if IMAGE_SIZE[0] < IMAGE_SIZE[1] else 2

    def vis_dataset(self, data_manager, label):
        """
        Visualize image and mask of given data set
        :param data_manager:
        :param label: the name you want to display in the event
        """
        print("Visualizing given data set, Image and Mask are concatenate and saving to TensorBoard event. \n"
              "Please use: tensorboard --logdir= PATH TO TENSORBOARD LOG DIR --samples_per_plugin=images=10000 in command line and open link in chrome or firefox explore")
        image_placeholder = tf.placeholder(tf.float32, shape=(data_manager.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], self.num_channel))
        mask_placeholder = tf.placeholder(tf.float32, shape=(data_manager.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], self.num_channel))
        image_mask_batch = tf.concat([image_placeholder, mask_placeholder], self.axis)
        #
        tensorboard_image = tf.summary.image(label, image_mask_batch, data_manager.num_batch * data_manager.batch_size)
        with self.session.as_default():
            for batch in range(data_manager.num_batch):
                # batch start
                img_batch, mask_batch, label_batch, _ = self.session.run(data_manager.next_batch)
                tensorboard_result = self.session.run(tensorboard_image, feed_dict={image_placeholder: img_batch,
                                                                                    mask_placeholder: mask_batch})

                self.writer.add_summary(tensorboard_result, batch)

    def vis_mask_out(self, data_manager, label, famliy):
        """
        Visualize the concatenated mask ou of the model using TensorBoard
        :param data_manager:
        """
        mask_out_batch = tf.concat([self.model.image_input, self.model.mask_out, self.model.mask], self.axis)
        # split and concat image in order to make it convenient to view
        if data_manager.batch_size > 1:
            splited_batch = tf.split(mask_out_batch, data_manager.batch_size, 0)
            mask_out_batch1 = tf.concat(splited_batch[:len(splited_batch) // 2], self.axis)
            mask_out_batch2 = tf.concat(splited_batch[len(splited_batch) // 2:], self.axis)
            mask_out_batch = tf.concat([mask_out_batch1, mask_out_batch2], 1 if self.axis == 2 else 1)

        tensorboard_image = tf.summary.image(label, mask_out_batch, data_manager.num_batch * data_manager.batch_size, family=famliy)
        for batch in range(data_manager.num_batch):

            img_batch, mask_in_batch, _, path = self.session.run(data_manager.next_batch)
            tensorboard_result = self.session.run(tensorboard_image, feed_dict={self.model.image_input: img_batch,
                                                                                self.model.mask: mask_in_batch,
                                                                                self.model.is_training_seg: False,
                                                                                self.model.is_training_dec: False})
            self.writer.add_summary(tensorboard_result, batch)

    def add_summary(self, summary, step):
        self.writer.add_summary(summary, step)

    def close(self):
        self.writer.close()
