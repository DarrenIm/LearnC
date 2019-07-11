#!/usr/bin/env python
# coding: utf-8

import matplotlib.image as mpimg
import tensorflow as tf
from MRI_preprocess import resampled_nii_generator
from preprocessing.data_utils import *
from network.loss import *
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

def predict():
    # X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="X")
    # y = tf.placeholder(tf.float32, shape=[None, 256, 256, num_classes], name="y")
    generator = resampled_nii_generator()
    num_classes = 40
    total_dice = []
    brain_section_count = []
    p = os.getcwd()
    path = os.path.join(p, 'saved_models')
    brain_name = []
    with open('/data/brain_segmentation/label_index_map_new.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split(',')
            brain_name.append(line[0])
    with open('/home/xiaodong/dice.csv', 'w') as out:
        out.write(','.join(['name'] + brain_name))
        out.write('\n')
        with tf.Session() as sess:
            start_time = time.time()
            saver = tf.train.import_meta_graph(path + '/73/model.ckpt.meta')
            saver.restore(sess, path + "/73/model.ckpt")
            X = tf.get_collection("inputs")[0]
            mode = tf.get_collection("inputs")[1]
            pred1 = tf.get_collection("outputs")

            for data, label, new_affine, filename in generator:
                slice_256 = []
                for i in range(0, data.shape[-1], 4):
                    slice_batch = []
                    for j in range(i, i+4):
                        slice_batch.append(normalize_data(data[:,:,j]))
                    X_test_op = np.expand_dims(np.stack(slice_batch, axis=0), axis=-1) # (4, 256, 256, 1)

                    pred = sess.run(pred1, feed_dict={X: X_test_op, mode: False})
                    pred_prob = tf.nn.softmax(pred[0], 3)  # 4, 256, 256, 40
                    prediction_np = sess.run(pred_prob)
                    prediction_np = np.transpose(np.argmax(prediction_np, axis=-1), (1,2,0)).astype(np.uint8) # 256, 256, 4
                    slice_256.append(prediction_np)

                np_case = np.concatenate(slice_256, axis=-1) # 256, 256, 256
                # print(np_case.shape, np_case.dtype)
                # new_image = nib.Nifti1Image(np_case, affine=new_affine)
                # nib.save(new_image, 'test_prediction.nii')
                # _image = nib.Nifti1Image(data, affine=new_affine)
                # nib.save(_image, 'test_input.nii')
                # _label = nib.Nifti1Image(label, affine=new_affine)
                # nib.save(_label, 'test_label.nii')
                pred_case = one_hot_encode(np_case, num_classes) # 256, 256, 256, 40
                label_case = one_hot_encode(label, num_classes)  # 256, 256, 256, 40
                # print(pred_case.shape)
                # print(label_case.shape)

                result = dice_score(pred_case, label_case)
                result = [str(x) for x in result]
                out.write(','.join(filename + result))
                out.write('\n')
                end_time = time.time()
                hours, rem = divmod(end_time - start_time, 3600)
                start_time = end_time
                minutes, seconds = divmod(rem, 60)
                print("-------------Completion of one case----------------")
                print("Total time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                print(result)
                total_dice.append(result)
                # brain_section_count.append(count_pixel_proba(pred_case))

    dice_matrix = np.stack(total_dice, axis=0)
    # calculate mean when the entries have non-zero values, may need full data for plotting
    nonzero_dice_mean = np.nanmean(np.where(dice_matrix!=0, dice_matrix, np.nan), 0)
    return nonzero_dice_mean, brain_section_count


if __name__ == '__main__':
    mean_dice, mean_brain = predict()
    with open('mean_dice_adni.pickle', 'wb') as f:
        pickle.dump(mean_dice, f, pickle.HIGHEST_PROTOCOL)
    with open('pixel_count_adni.pickle', 'wb') as f:
        pickle.dump(mean_brain, f, pickle.HIGHEST_PROTOCOL)
    print('mean dice of every label:', mean_dice)
    print('brain pixel count of every label:', mean_brain)