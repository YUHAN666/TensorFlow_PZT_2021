import time

import tensorflow as tf
from tqdm import tqdm

from config import IMAGE_SIZE, DATA_FORMAT, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_VALID
from model.loss_func.balanced_crossentropy import balanced_crossentropy_with_logits
from utiles.iouEval import iouEval


class Trainer(object):

    def __init__(self, sess, model, param, logger, tensorboard_manager):
        self.session = sess
        self.model = model
        self.learning_rate = param["learning_rate"]
        self.optimizer = param["optimizer"]
        self.epochs = param["epochs"]
        self.save_frequency = param["save_frequency"]
        self.mode = param["mode"]
        self.loss = param["loss"]
        self.lr_decay = param["lr_decay"]
        self.decay_rate = param["decay_rate"]
        self.decay_steps = param["decay_steps"]
        self.staircase = param["stair_case"]
        self.check_seg_frequency = param["check_seg_frequency"]
        self.balanced_mode = param["balanced_mode"]
        self.logger = logger
        self.tensorboard_manager = tensorboard_manager

        with self.session.as_default():

            if self.lr_decay:
                self.global_step = tf.Variable(0, trainable=False)
                self.add_global = self.global_step.assign_add(1)
                self.learning_rate = self.learning_rate_decay()
            self.summary_learning_rate = tf.summary.scalar("learning_rate", self.learning_rate)

            if self.mode == "train_segmentation":
                train_segment_var_list = [v for v in tf.trainable_variables() if ('backbone' in v.name) or ('segmentation' in v.name)]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                update_ops_segment = [v for v in update_ops if ('backbone' in v.name) or ('segmentation' in v.name)]
                optimizer_segmentation = self.optimizer_func()
                segmentation_loss = self.segmentation_loss_func(self.model.segmentation_output, self.model.mask)
                with tf.control_dependencies(update_ops_segment):
                    optimize_segment = optimizer_segmentation.minimize(segmentation_loss, var_list=train_segment_var_list)
                self.segmentation_loss = segmentation_loss
                self.optimize_segment = optimize_segment
                self.summary_segmentation_loss_train = tf.summary.scalar("segmentation_loss_train", self.segmentation_loss)
                self.summary_segmentation_loss_valid = tf.summary.scalar("segmentation_loss_valid", self.segmentation_loss)

            elif self.mode == "train_decision":
                train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                update_ops_decision = [v for v in update_ops if 'decision' in v.name]
                optimizer_decision = self.optimizer_func()
                decision_loss = self.decision_loss_func(self.model.decision_output, self.model.label)
                with tf.control_dependencies(update_ops_decision):
                    optimize_decision = optimizer_decision.minimize(decision_loss, var_list=train_decision_var_list)
                self.decision_loss = decision_loss
                self.optimize_decision = optimize_decision
                self.summary_decision_loss_train = tf.summary.scalar("decision_loss_train", self.decision_loss)
                self.summary_decision_loss_valid = tf.summary.scalar("decision_loss_valid", self.decision_loss)

    def optimizer_func(self):

        if self.optimizer == "Adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == 'GD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'RMS':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer {}".format(self.optimizer))

        return optimizer

    def learning_rate_decay(self):

        if self.lr_decay == "exponential_decay":
            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
                                                       decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                       staircase=self.staircase)
        elif self.lr_decay == "inverse_time_decay":
            # decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
            learning_rate = tf.train.inverse_time_decay(self.learning_rate, global_step=self.global_step,
                                                        decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                        staircase=self.staircase)
        elif self.lr_decay == "natural_exp_decay":
            # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step / decay_steps)
            learning_rate = tf.train.natural_exp_decay(self.learning_rate, global_step=self.global_step,
                                                       decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                       staircase=self.staircase)
        else:
            raise ValueError("Unsupported learning rate decay strategy {}".format(self.lr_decay))

        return learning_rate

    def segmentation_loss_func(self, segmentation_output, mask):
        """ Segmentation loss"""
        if len(segmentation_output) == 5:
            if DATA_FORMAT == 'channels_first':
                for nec_index in range(len(segmentation_output)):
                    segmentation_output[nec_index] = tf.transpose(segmentation_output[nec_index], [0, 2, 3, 1])
            logits_pixel = tf.image.resize_images(segmentation_output[0], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True,
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if self.balanced_mode:
                loss = balanced_crossentropy_with_logits
            else:
                loss = tf.nn.sigmoid_cross_entropy_with_logits

            segmentation_loss = tf.reduce_mean(loss(logits=logits_pixel, labels=mask)) + \
                                tf.reduce_mean(loss(logits=segmentation_output[1], labels=tf.image.resize_images(mask, (
                                IMAGE_SIZE[0] // 4, IMAGE_SIZE[1] // 4)))) + \
                                tf.reduce_mean(loss(logits=segmentation_output[2], labels=tf.image.resize_images(mask, (
                                IMAGE_SIZE[0] // 8, IMAGE_SIZE[1] // 8)))) + \
                                tf.reduce_mean(loss(logits=segmentation_output[3], labels=tf.image.resize_images(mask, (
                                IMAGE_SIZE[0] // 16, IMAGE_SIZE[1] // 16)))) + \
                                tf.reduce_mean(loss(logits=segmentation_output[4], labels=tf.image.resize_images(mask, (
                                IMAGE_SIZE[0] // 32, IMAGE_SIZE[1] // 32))))

        elif len(segmentation_output) == 2:
            if DATA_FORMAT == 'channels_first':
                for nec_index in range(len(segmentation_output)):
                    segmentation_output[nec_index] = tf.transpose(segmentation_output[nec_index], [0, 2, 3, 1])
            logits_pixel = tf.image.resize_images(segmentation_output[1], (IMAGE_SIZE[0], IMAGE_SIZE[1]), align_corners=True)
            segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=mask))

        else:
            raise ValueError("Wrong number of segmentation output")

        return segmentation_loss

    def decision_loss_func(self, dec_out, label):
        """ Decision loss"""
        if self.loss == "cross_entropy":
            decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)
            decision_loss = tf.reduce_mean(decision_loss)
        else:
            raise ValueError("Unsupported decision loss {}".format(self.loss))
        return decision_loss

    def train_segmentation(self, data_manager_train, data_manager_valid, saver):
        """ Train the segmentation part of the model """
        self.logger.info("Start training segmentation for {} epochs, {} steps per epochs, batch size is {}. Save to checkpoint every {} epochs "
                         .format(self.epochs, data_manager_train.num_batch, data_manager_train.batch_size, self.save_frequency))
        self.logger.info("Loss: {}, Optimizer: {}, Learning_rate: {}".format(self.loss, self.optimizer, self.learning_rate))
        if self.lr_decay:
            self.logger.info("Using {} strategy, decay_rate: {}， decay_steps: {}, staircase: {}".format(self.lr_decay, self.decay_rate, self.decay_steps, self.staircase))
        current_epoch = saver.step + 1
        with self.session.as_default():
            print('Start training segmentation for {} epochs, {} steps per epoch'.format(self.epochs, data_manager_train.num_batch))
            tensorboard_merged = tf.summary.merge([self.summary_learning_rate, self.summary_segmentation_loss_train])
            trainIoU = iouEval(data_manager_train.batch_size)
            train_loss = []
            train_iou = []
            val_loss = []
            val_iou = []
            iter_loss = []
            train_iou_placeholder = tf.placeholder(tf.float32)  # placeholder for TensorBoard
            val_iou_placeholder = tf.placeholder(tf.float32)
            summary_train_iou = tf.summary.scalar("train set iou", train_iou_placeholder)
            summary_val_iou = tf.summary.scalar("val set iou", val_iou_placeholder)
            for i in range(current_epoch, self.epochs+current_epoch):
                trainIoU.reset()
                print('Epoch {}:'.format(i))
                time.sleep(0.1)
                pbar = tqdm(total=data_manager_train.num_batch, leave=True)
                # epoch start
                for batch in range(data_manager_train.num_batch):
                    # batch start

                    img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_train.next_batch)

                    mask_in, mask_out, _, loss_value_batch, tensorboard_result = self.session.run([self.model.mask,
                                                                                                   self.model.mask_out,
                                                                                                   self.optimize_segment,
                                                                                                   self.segmentation_loss,
                                                                                                   tensorboard_merged],
                                                              feed_dict={self.model.image_input: img_batch,
                                                                         self.model.mask: mask_batch,
                                                                         self.model.label: label_batch,
                                                                         self.model.is_training_seg: TRAIN_MODE_IN_TRAIN,
                                                                         self.model.is_training_dec: False})
                    self.tensorboard_manager.add_summary(tensorboard_result, (i - 1) * data_manager_train.num_batch + batch)
                    trainIoU.addBatch(mask_in, mask_out)
                    iter_loss.append(loss_value_batch)
                    pbar.update(1)
                    if self.lr_decay:
                        _, lr = self.session.run([self.add_global, self.learning_rate])
                pbar.clear()
                pbar.close()
                time.sleep(0.1)
                # loss and iou check
                train_loss.append(sum(iter_loss)/len(iter_loss))
                train_iou.append(trainIoU.getIoU())
                val_loss_epo, val_iou_epo = self.valid_segmentation(data_manager_valid, i)
                val_loss.append(val_loss_epo)
                val_iou.append(val_iou_epo)
                self.logger.info("Epoch{}  train_loss:{}, train_iou:{}, val_loss:{}, val_iou:{}"
                                 .format(i, iter_loss[i-current_epoch], train_iou[i-current_epoch], val_loss[i-current_epoch], val_iou[i-current_epoch]))
                print('train_loss:{}, train_iou:{},  val_loss:{}, val_iou:{}'
                      .format(iter_loss[i-current_epoch], train_iou[i-current_epoch], val_loss[i-current_epoch], val_iou[i-current_epoch]))

                # export to TensorBoard
                tensorboard_train_iou, tensorboard_val_iou = self.session.run([summary_train_iou, summary_val_iou],
                                                                              feed_dict={train_iou_placeholder: trainIoU.getIoU(),
                                                                                         val_iou_placeholder: val_iou_epo})
                self.tensorboard_manager.add_summary(tensorboard_train_iou, i)
                self.tensorboard_manager.add_summary(tensorboard_val_iou, i)

                if (i-current_epoch+1) % self.save_frequency == 0 or i == self.epochs + current_epoch:
                    # if val_loss < best_loss:
                    # best_loss = val_loss
                    # print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
                    saver.save_checkpoint(i)

                if (i-current_epoch+1) % self.check_seg_frequency == 0 or i == self.epochs + current_epoch:
                    self.logger.info("Writing concatenated mask_out into TensorBoard event. \nTo view it, "
                                     "use --logdir= PATH TO TENSORBOARD LOG DIR --samples_per_plugin=images=10000 in command line and open link in chrome or firefox explore")
                    print("Writing concatenated mask_out into TensorBoard event. \nTo view it, "
                          "use --logdir= PATH TO TENSORBOARD LOG DIR --samples_per_plugin=images=10000 in command line and open link in chrome or firefox explore")
                    self.tensorboard_manager.vis_mask_out(data_manager_train, "train mask out", "Epoch{}".format(i))
                    self.tensorboard_manager.vis_mask_out(data_manager_valid, "valid mask out", "Epoch{}".format(i))

        self.logger.info("Complete training segmentation, reduce train_loss from {} to {}, increase train_iou from {} to {} "
                         "reduce val_loss from {} to {}, increase val_iou from {} to {}"
                         .format(train_loss[0], train_loss[-1], train_iou[0], train_iou[-1], val_loss[0], val_loss[-1], val_iou[0], val_iou[-1]))

    def valid_segmentation(self, data_manager_valid, epoch):
        """ Evaluate the segmentation part during training process"""
        with self.session.as_default():
            # print('start validating segmentation')
            total_loss = 0.0
            num_step = 0.0
            valIoU = iouEval(data_manager_valid.batch_size)

            for batch in range(data_manager_valid.num_batch):
                img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_valid.next_batch)

                mask_in, mask_out, total_loss_value_batch, tensorboard_result = self.session.run([self.model.mask,
                                                                                                  self.model.mask_out,
                                                                                                  self.segmentation_loss,
                                                                                                  self.summary_segmentation_loss_valid],
                                                             feed_dict={self.model.image_input: img_batch,
                                                                        self.model.mask: mask_batch,
                                                                        self.model.label: label_batch,
                                                                        self.model.is_training_seg: TRAIN_MODE_IN_VALID,
                                                                        self.model.is_training_dec: TRAIN_MODE_IN_VALID})
                self.tensorboard_manager.add_summary(tensorboard_result, epoch)
                valIoU.addBatch(mask_in, mask_out)
                num_step = num_step + 1
                total_loss += total_loss_value_batch

            total_loss = total_loss/num_step
            val_iou = valIoU.getIoU()
            return total_loss, val_iou

    def train_decision(self, data_manager_train, data_manager_valid, saver):
        """Train the decision part of model"""
        with self.session.as_default():
            print('Start training decision for {} epochs, {} steps per epoch'.format(self.epochs, data_manager_train.num_batch))
            self.logger.info("Start training decision for {} epochs, {} steps per epochs, batch size is {}. Save to checkpoint every {} epochs "
                .format(self.epochs, data_manager_train.num_batch, data_manager_train.batch_size, self.save_frequency))
            self.logger.info("Loss: {}, Optimizer: {}, Learning_rate: {}".format(self.loss, self.optimizer, self.learning_rate))
            if self.lr_decay:
                self.logger.info("Using {} strategy, decay_rate: {}， decay_steps: {}, staircase: {}"
                                 .format(self.lr_decay, self.decay_rate, self.decay_steps, self.staircase))
            current_epoch = saver.step + 1
            tensorboard_merged = tf.summary.merge([self.summary_learning_rate, self.summary_decision_loss_train])
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []
            iter_loss = []
            for i in range(current_epoch, self.epochs+current_epoch):
                print('Epoch {}:'.format(i))
                time.sleep(0.1)
                pbar = tqdm(total=data_manager_train.num_batch, leave=True)
                # epoch start
                true_account = 0
                false_account = 0
                for batch in range(data_manager_train.num_batch):
                    # batch start
                    img_batch, mask_batch, label_batch, path = self.session.run(data_manager_train.next_batch)

                    decision_out, _, loss_value_batch, tensorboard_result = self.session.run([self.model.decision_out,
                                                                                              self.optimize_decision,
                                                                                              self.decision_loss,
                                                                                              tensorboard_merged],
                                                                            feed_dict={self.model.image_input: img_batch,
                                                                                       self.model.mask: mask_batch,
                                                                                       self.model.label: label_batch,
                                                                                       self.model.is_training_seg: False,
                                                                                       self.model.is_training_dec: TRAIN_MODE_IN_TRAIN})
                    self.tensorboard_manager.add_summary(tensorboard_result, (i-1)*data_manager_train.num_batch+batch)

                    iter_loss.append(loss_value_batch)
                    pbar.update(1)
                    for b in range(data_manager_train.batch_size):
                        if (decision_out[b] > 0.5 and label_batch[b] == 1) or (decision_out[b] < 0.5 and label_batch[b] == 0):
                            true_account += 1
                        else:
                            false_account += 1

                    if self.lr_decay:
                        _, lr = self.session.run([self.add_global, self.learning_rate])
                train_acc.append(true_account/(true_account + false_account))
                pbar.clear()
                pbar.close()
                train_loss.append(sum(iter_loss)/len(iter_loss))

                time.sleep(0.1)
                val_loss_epo, val_acc_epo = self.valid_decision(data_manager_valid, i)
                val_loss.append(val_loss_epo)
                val_acc.append(val_acc_epo)
                print('train_loss:{}, train_accuracy:{}  val_loss:{}, val_accuracy:{}'.
                      format(train_loss[i-current_epoch], train_acc[i-current_epoch], val_loss[i-current_epoch], val_acc[i-current_epoch]))
                self.logger.info("Epoch {}  train_loss: {}, train_accuracy: {}, val_loss: {}, val_accuracy: {}"
                                 .format(i+saver.step, train_loss[i-current_epoch], train_acc[i-current_epoch], val_loss[i-current_epoch], val_acc[i-current_epoch]))

                if i % self.save_frequency == 0 or i == self.epochs:
                    # if val_loss < best_loss:
                    # best_loss = val_loss
                    # print('reduce loss to {}, saving model at epoch:{}'.format(val_loss, i))
                    saver.save_checkpoint(i)
        self.logger.info("Complete training decision, reduce train_loss from {} to {}, increase train_accuracy from {} to {},  "
                         "reduce val_loss from {} to {}, increase val_accuracy from {} to {}"
                         .format(train_loss[0], train_loss[-1], train_acc[0], train_acc[-1], val_loss[0], val_loss[-1], val_acc[0], val_acc[-1]))

    def valid_decision(self, data_manager_valid, epoch):
        """ Evaluate the performance of decision part during training"""
        with self.session.as_default():
            # print('start validating decision')
            total_loss = 0.0
            num_step = 0.0
            true_account = 0
            false_account = 0
            for batch in range(data_manager_valid.num_batch):
                img_batch, mask_batch, label_batch, _ = self.session.run(data_manager_valid.next_batch)

                decision_out, total_loss_value_batch, tensorboard_result = self.session.run([self.model.decision_out,
                                                                                             self.decision_loss,
                                                                                             self.summary_decision_loss_valid],
                                                             feed_dict={self.model.image_input: img_batch,
                                                                        self.model.mask: mask_batch,
                                                                        self.model.label: label_batch,
                                                                        self.model.is_training_seg: TRAIN_MODE_IN_VALID,
                                                                        self.model.is_training_dec: TRAIN_MODE_IN_VALID})
                self.tensorboard_manager.add_summary(tensorboard_result, epoch)
                for b in range(data_manager_valid.batch_size):
                    if (decision_out[b] > 0.5 and label_batch[b] == 1) or (decision_out[b] < 0.5 and label_batch[b] == 0):
                        true_account += 1
                    else:
                        false_account += 1
                num_step = num_step + 1
                total_loss += total_loss_value_batch
            accuracy = true_account/(true_account+false_account)
            total_loss /= num_step
            return total_loss, accuracy
