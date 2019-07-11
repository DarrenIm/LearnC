#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from network.loss import *
from network.quickNAT import quick_nat
from preprocessing.data_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# list path

# valid_zeros_dir = '/data/tfrecords_png/zeros_valid.list'
# train_zeros_dir = '/data/tfrecords_png/zeros_train.list'

train_nonzeros_dir = '/data/tfrecords/sagittal/nonzeros_train.list'
valid_nonzeros_dir = '/data/tfrecords/sagittal/nonzeros_valid.list'

# load list and files
with open(train_nonzeros_dir) as file:
    train_list = list(file)
with open(valid_nonzeros_dir) as file2:
    valid_list = list(file2)
# with open(train_zeros_dir) as file:
#     train_zeros_list = list(file)
# with open(valid_zeros_dir) as file:
#     valid_zeros_list = list(file)

# append nonzero list with 0.5*zero list
# index = 66
# train_num = int(0.5 * len(train_nonzeros_list))
# valid_num = int(0.5 * len(valid_nonzeros_list))
# random.Random(index).shuffle(train_zeros_list)
# random.Random(index).shuffle(valid_zeros_list)
# train_nonzeros_list.extend(train_zeros_list[0:train_num])
# valid_nonzeros_list.extend(valid_zeros_list[0:valid_num])
# train_list = train_nonzeros_list
# valid_list = valid_nonzeros_list
num_train_examples = len(train_list)
num_val_examples = len(valid_list)

with open('./saved_pickles/label_distribution_new.pickle', 'rb') as f:
    label_distribution = pickle.load(f)
with open('./saved_pickles/map_dictionary.pickle', 'rb') as f1:
    map_dict = pickle.load(f1)

# config
num_classes = 40
epochs = 5
batch_size = 5  # max
display_step = 10
n_train = num_train_examples
n_valid = num_val_examples
train_total_batch = int(n_train / batch_size)
val_total_batch = int(n_valid / batch_size)
train_logs_path = "logs/train"
val_logs_path = "logs/val"
learning = 0.001
# learning = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)
momentum = 0.9
nestrov = True
ckdir = "saved_models/model.ckpt"
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)  # GPU Configuration


def train(restore=False, testing=False):
    # log directory of graphs
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(train_logs_path, "train_deep_sdnet", current_time)
    test_logdir = os.path.join(val_logs_path, "test_deep_sdnet", current_time)

    # train_dataset
    X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="X")
    y = tf.placeholder(tf.float32, shape=[None, 256, 256, num_classes], name="y")

    mode = tf.placeholder(tf.bool, name="mode")
    pred1 = quick_nat(X, mode, num_classes)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred1)

    pred_prob = tf.nn.softmax(pred1, 3)

    with tf.name_scope('loss_cross_entropy'):
        # Build loss
        # loss_op_1 = tf.losses.sigmoid_cross_entropy(y, logits=pred1)
        # loss_op_1 = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred1)
        # loss_op_1 = weighted_cross_entropy(pred1, y, num_classes)
        loss_op_1 = weighted_cross_entropy(pred1, y, label_distribution)
        tf.summary.scalar("Loss_cross_entropy", loss_op_1)

    with tf.name_scope('loss_dice_background'):
        loss_op_2 = dice_coef_axis(pred_prob, y, 0)
        tf.summary.scalar("Loss_Dice_Background", loss_op_2)

    with tf.name_scope('loss_dice_nonzero_mean'):
        loss_op_3 = dice_coef_mean(pred_prob, y, num_classes)
        tf.summary.scalar("Loss_dice_nonzero_mean", loss_op_3)

    with tf.name_scope('IOU'):
        iou_op = IOU(pred_prob, y)
        tf.summary.scalar("Mean_IOU", iou_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    with tf.name_scope('SGD'):
        # Gradient Descent
        train_op_1 = make_train_op(pred1, y, learning, update_ops, num_classes, label_distribution)
        #train_op_1 = weighted_cross_entropy_plus_sum_dice(pred1, y, label_distribution)

    # with tf.name_scope('Accuracy'):
    #     acc = tf.equal(tf.round(tf.nn.sigmoid(pred1)), tf.round(y))
    #     acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    #     # acc = tf.Print(acc, [acc], message="accuracy: ")
    #     tf.summary.scalar('Accuracy', acc)

    prediction = tf.reshape(tf.cast(tf.argmax(pred1, axis=3), tf.float32), shape=[batch_size, 256, 256, 1])
    ground_truth = tf.reshape(tf.cast(tf.argmax(y, axis=3), tf.float32), shape=[batch_size, 256, 256, 1])

    TP = tf.count_nonzero(prediction * ground_truth, dtype=tf.float32)
    TN = tf.count_nonzero((prediction - 1) * (ground_truth - 1), dtype=tf.float32)
    FP = tf.count_nonzero(prediction * (ground_truth - 1), dtype=tf.float32)
    FN = tf.count_nonzero((prediction - 1) * ground_truth, dtype=tf.float32)

    with tf.name_scope('precision'):
        precision = TP / (TP + FP)
        # tf.Print(precision, [precision], message="Precision: ")
        tf.summary.scalar('precision', precision)

    with tf.name_scope('recall'):
        recall = TP / (TP + FN)
        # tf.Print(recall, [recall], message="Recall: ")
        tf.summary.scalar('recall', recall)

    with tf.name_scope('FPR'):
        fallout = FP / (FP + TN)
        tf.summary.scalar('False Positive Rate', fallout)

    with tf.name_scope('F1_score'):
        f1_score = (2 * (precision * recall)) / (precision + recall)
        tf.summary.scalar('F1 score', f1_score)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("./saved_model")
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        # create log writer object
        train_summary_writer = tf.summary.FileWriter(train_logdir, graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)
        global_step = tf.train.get_global_step(sess.graph)
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        seed = 8

        for epoch in range(epochs):
            random.Random(seed).shuffle(train_list)
            random.Random(seed).shuffle(valid_list)
            # training
            step_count_train = int(len(train_list) / batch_size)  # total iteration for one sample
            #step_count_train = 1000
            idx = 0
            iou_sum = 0
            dice_sum = 0
            loss_sum = 0
            dice_0 = 0
            print("-----------training---------------")
            for i in range(step_count_train): # in one iteration
                    X_batch_op, y_batch_op = data_generator(batch_size, train_list[idx: idx+batch_size], map_dict, num_classes)

                    _, step_loss_train, step_dice_0, step_dice_mean, step_iou, step_summary, global_step_value = sess.run(
                        [train_op_1, loss_op_1, loss_op_2, loss_op_3, iou_op, summary_op, global_step],
                        feed_dict={X: X_batch_op,
                                   y: y_batch_op,
                                   mode: True})

                    # sum up
                    iou_sum += step_iou
                    loss_sum += step_loss_train  # loss cross-entropy
                    dice_sum += step_dice_mean # non-zero mean dice
                    dice_0 += step_dice_0

                    # Display logs per display_step and calculate mean logs
                    if (i + 1) % display_step == 0:
                        # write logs
                        train_summary_writer.add_summary(step_summary, epoch)
                        # mean
                        mean_iou = iou_sum/display_step
                        mean_loss_train = loss_sum/display_step
                        mean_dice = dice_sum/display_step
                        mean_dice_0 = dice_0/display_step
                        print("epoch:", epoch, " step: ", i + 1, "/", step_count_train)
                        print("Dice_nonzero_mean=", "{:.9f}".format(mean_dice))
                        print("Dice_background_mean=", "{:.9f}".format(mean_dice_0))
                        print("loss_cross_entropy=", "{:.9f}".format(mean_loss_train))
                        print("mean_IOU=", "{:.9f}".format(mean_iou))
                        # iou_sum = dice_sum = loss_sum = dice_0 = 0
                        iou_sum = 0
                        loss_sum = 0
                        dice_sum = 0
                        dice_0 = 0

                    idx += batch_size
            # saver.save(sess, ckdir, global_step=epoch)
            # print("Model saved in file: %s" % ckdir)

            # validation
            step_count_valid = int(len(valid_list) / batch_size)
            #step_count_valid = 200
            idx = 0
            iou_sum = 0
            dice_sum = 0
            loss_sum = 0
            print("-----------validation-------------")
            for i in range(step_count_valid):
                    X_valid_op, y_valid_op = data_generator(batch_size, train_list[idx: idx+batch_size], map_dict, num_classes)

                    step_loss_valid, loss_dice_mean, step_iou, step_summary = sess.run(
                        [loss_op_1, loss_op_3, iou_op, summary_op],
                        feed_dict={X: X_valid_op,
                                   y: y_valid_op,
                                   mode: False})

                    # sum
                    loss_sum += step_loss_valid
                    dice_sum += loss_dice_mean
                    iou_sum += step_iou

                    # Display logs per display_step
                    if (i + 1) % display_step == 0:
                        test_summary_writer.add_summary(step_summary, epoch)
                        # mean
                        mean_loss_valid = loss_sum/display_step
                        mean_dice = dice_sum/display_step
                        mean_iou = iou_sum/display_step

                        print("epoch:", epoch, " step: ", i + 1, "/", step_count_valid)
                        print("loss_cross_entropy=", "{:.9f}".format(mean_loss_valid))
                        print("Dice_nonzero=", "{:.9f}".format(mean_dice))
                        print("mean_IOU=", "{:.9f}".format(mean_iou))
                        iou_sum = 0
                        dice_sum = 0
                        loss_sum = 0

                    idx += batch_size
            seed += 666  # complete of one epoch
        train_summary_writer.close()
        test_summary_writer.close()


if __name__ == '__main__':
    train()


# ### Load Test Image

# In[11]:


# test_data, test_labels = read_dataset(filepath=test_filepath, file_for="test")
#
#
# # In[12]:
#
#
# test_data = test_data.reshape(test_data.shape[0], test_data.shape[2], test_data.shape[3], test_data.shape[1])
#
#
# # In[13]:
#
#
# plt.imshow(test_data[100].reshape(256,256))
#
#
# # In[14]:
#
#
# test_liver_spleen_original = split_dataset(test_labels, dataset="liver_spleen")
#
#
# # In[15]:
#
#
# plt.imshow(test_liver_spleen_original[100])
#
#
# # ### Load saved Model
#
# # In[16]:
#
#
# sess = tf.Session()
#
#
# # In[17]:
#
#
# sess.run(tf.global_variables_initializer())
# saver = tf.train.import_meta_graph("./saved_models_liver_spleen/model.ckpt-")
# saver.restore(sess, tf.train.latest_checkpoint("./saved_models_liver_spleen"))
# X = tf.get_collection("inputs")[0]
# mode = tf.get_collection("inputs")[1]
#
#
# # In[18]:
#
#
# pred1 = tf.get_collection("outputs")[0]
#
#
# # In[19]:
#
#
# test_data[100].shape
#
#
# # In[20]:
#
#
# test_image = centeredCrop((test_data[100]).reshape(1,256,256,1),192,192)
#
#
# # In[21]:
#
#
# pred = sess.run(pred1, feed_dict={X: test_image , mode: False})
#
#
# # In[22]:
#
#
# plt.imshow(np.argmax(pred[0],2))
#
#
# # In[23]:
#
#
# pred_np = sess.run(tf.nn.softmax(pred))
#
#
# # In[ ]:




