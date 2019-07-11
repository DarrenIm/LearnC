#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import matplotlib.image as mpimg
import tensorflow as tf
from MRI_preprocess import resampled_nii_generator
import time
from preprocessing.data_utils import *
from network.loss import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
p = os.getcwd()
path = os.path.join(p, 'saved_models')
num_classes = 40
smooth = 1e-7

def dice_score(img, label):
    dice_current = np.zeros(img.shape[-1])
    for i in range(img.shape[-1]):
        y_true_f = np.reshape(label[:, :, :, i], (1,-1))
        y_pred_f = np.reshape(img[:, :, :, i], (1,-1))
        intersection = np.sum(y_true_f * y_pred_f)
        denominator = np.sum(y_true_f) + np.sum(y_pred_f)
        dice_current[i] = 2 * intersection / (denominator + smooth)
    return dice_current

def count_pixel_proba(img):
    current = np.zeros(img.shape[-1])
    for i in range(img.shape[-1]):
        pixel_number = np.sum(img[:, :, :, i])
        current[i] = pixel_number
    return current

def argmax_pixel_count(img):
    current_a = np.zeros(img.shape[-1])
    for i in range(img.shape[-1]):
        pixel_number = np.count_nonzero((np.argmax(img, axis=3)))
        current_a[i] = pixel_number
    return current_a

def data_generator():
    list_path = '/data/tfrecords/frontal/test/infer_data.list'
    with open(list_path) as f:
        testfilenames = list(f)
    for filename in testfilenames:
        img = Image.open(os.path.join('/data/tfrecords/frontal/test/image', filename.strip()))
        label = Image.open(os.path.join('/data/tfrecords/frontal/test/label', filename.strip()))
        yield img, label

def predict():

    generator = data_generator()
    i = 0
    total_dice = []
    brain_section_count = []
    with tf.Session() as sess:
        start_time = time.time()
        saver = tf.train.import_meta_graph(path + '/73/model.ckpt.meta')
        saver.restore(sess, path + "/73/model.ckpt")
        X = tf.get_collection("inputs")[0]
        mode = tf.get_collection("inputs")[1]
        pred1 = tf.get_collection("outputs")
        slice_256, label_256 = [], []
        for data, label in generator:
            assert not np.any(np.isnan(data))
            assert not np.any(np.isnan(label))
            # for i in range(data.shape[-1]):
            X_test_op = np.expand_dims(normalize_data(data), axis=0)
            X_test_op = np.expand_dims(X_test_op, axis=-1) # (1, 256, 256, 1)
            y_test_op = np.expand_dims(one_hot_encode(label, num_classes), axis=0) # (1, 256, 256, 40)

            pred = sess.run(pred1, feed_dict={X: X_test_op, mode: False})
            pred_prob = tf.nn.softmax(pred[0], 3)
            prediction_np = sess.run(pred_prob) # 1, 256, 256, 40

            prediction_np, y_test_op = np.squeeze(prediction_np, axis=0), np.squeeze(y_test_op, axis=0)
            slice_256.append(np.argmax(prediction_np, axis=-1))
            label_256.append(y_test_op)
            i += 1

            if i % 256 == 0:
                pred_case = one_hot_encode(np.stack(slice_256, axis=0), num_classes) # 256, 256, 256, 40
                label_case = np.stack(label_256, axis=0)
                slice_256, label_256 = [], []
                # append lists of [1, num_classes] np array
                print(pred_case.shape)
                result = dice_score(pred_case, label_case)
                end_time = time.time()
                hours, rem = divmod(end_time - start_time, 3600)
                start_time = end_time
                minutes, seconds = divmod(rem, 60)
                print("-------------Completion of one case----------------")
                print("Total training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                print(result)
                total_dice.append(result)
                brain_section_count.append(count_pixel_proba(prediction_np))

    dice_matrix = np.stack(total_dice)
    section_prob_matrix = np.stack(brain_section_count)
    # calculate mean when the entries have non-zero values
    nonzero_dice_mean = np.nanmean(np.where(dice_matrix!=0, dice_matrix, np.nan), 0)
    nonzero_prob_mean = np.nanmean(np.where(section_prob_matrix!=0, section_prob_matrix, np.nan), 0)
    return nonzero_dice_mean, nonzero_prob_mean


if __name__ == '__main__':
    mean_dice, mean_brain = predict()
    with open('mean_dice_adni.pickle', 'wb') as f:
        pickle.dump(mean_dice, f, pickle.HIGHEST_PROTOCOL)
    with open('pixel_count_adni.pickle', 'wb') as f:
        pickle.dump(mean_brain, f, pickle.HIGHEST_PROTOCOL)
    print('mean dice of every label:', mean_dice)
    print('brain pixel count of every label:', mean_brain)