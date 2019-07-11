import tensorflow as tf
import statistics
import numpy as np
import operator
import random
import operator
import collections
smooth = 1e-7
import time

fre = [9.29231716e-01,1.32689618e-02,1.49098567e-02,4.87518542e-04,
         2.24650656e-05,8.05163354e-04,3.25848809e-03,4.67987337e-04,
         2.16884838e-04,3.03452039e-04,1.21072012e-04,6.10418773e-05,
         1.01665779e-04,1.21584256e-03,2.45323269e-04,9.62539202e-05,
         6.08052649e-05,3.31334738e-05,2.49399116e-04,3.34514512e-05,
         1.32584995e-02,1.49245817e-02,4.34995398e-04,2.22669978e-05,
         7.75046814e-04,3.26791430e-03,4.51200833e-04,2.25293332e-04,
         3.05796586e-04,1.16752635e-04,2.51648893e-04,1.04385095e-04,
         3.53312153e-05,2.48418312e-04,3.61687521e-05,6.00093322e-05,
         3.51675268e-05,3.88228360e-05,3.78938956e-05,5.57551136e-05]

def compute_mean_iou(total_cm, num_classes, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    condition = np.zeros((num_classes, num_classes))
    ones = np.ones((num_classes - 1, num_classes - 1))
    condition[1:, 1:] = ones
    total_cm = tf.where(condition, total_cm, tf.zeros_like(total_cm))
    # with tf.name_scope('total_cm'):
    #     total_cm = tf.where(condition, total_cm, tf.zeros_like(total_cm))
    #     total_cm = tf.Print(total_cm,[tf.reduce_sum(total_cm)], message="total_cm: ")
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    for i in range(1, num_classes):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result

def IOU(y_pred, y_true):
    """Returns a (approx) IOU score
    IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, num_classes = y_pred.get_shape().as_list()[1:]

    y_true_f = tf.reshape(tf.argmax(y_true, axis=3), [-1])
    y_pred_f = tf.reshape(tf.argmax(y_pred, axis=3), [-1])

    iou_ = tf.metrics.mean_iou(y_true_f, y_pred_f, num_classes)
    iou = compute_mean_iou(iou_[1], num_classes, name='mean_iou')
    return iou

# %%CLASS-WISE-DICE
def dice_coef_axis(y_pred, y_true, i):
    y_true_f = tf.reshape(y_true[:, :, :, i], [-1])
    y_pred_f = tf.reshape(y_pred[:, :, :, i], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    # return dice_coef


def dice_coef_0(y_pred, y_true):
    return dice_coef_axis(y_pred, y_true, 0)


def dice_coef_mean(y_pred, y_true, num_classes):
    dice = 0
    j = 0
    for i in range(1, num_classes):
        if dice_coef_axis(y_pred, y_true, i) == 0:
            pass
        else:
            dice += dice_coef_axis(y_pred, y_true, i)
            j += 1
    return dice/(j + smooth)


def dice_coef_2(y_true, y_pred):
    return dice_coef_axis(y_pred, y_true, 2)

def dice_coef_3(y_true, y_pred):
    return dice_coef_axis(y_pred, y_true, 3)


def compute_cross_entropy_mean(y_pred, y_true):
    # https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
    total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(tf.nn.softmax(y_pred)), [1]))
    return total_loss


def weighted_cross_entropy(y_pred, y_true, edge_map):
    '''
    @@@ From https://github.com/kwotsin/TensorFlow-ENet/blob/master/train_enet.py#L103
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want.
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.
    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------
    INPUTS:
    - y_pred(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - y_true(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    # class_weights_list = []
    # for k, v in distribution_dict.items():
    #     class_weights_list.append(v)
    # get median value
    weight_list_sorted = sorted(fre)
    median_value = statistics.median(weight_list_sorted)
    # class weights
    f = [median_value / v for v in fre]
    f[0] = f[0]*150
    # sorted_weights = collections.OrderedDict(sorted(class_weights_dict.items()))

    # normalizing
    # norm_weight_list = [float(i) / sum(class_weights_list) for i in class_weights_list]
    # norm_weight_list = [float(i) * 40 for i in norm_weight_list]
    sample_weights = tf.reduce_sum(tf.multiply(y_true, f), -1) + tf.multiply(edge_map, 2*median_value/min(f))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred, weights=sample_weights)
    return loss


def weighted_cross_entropy_plus_dice(y_pred, y_true, edge_map):
    loss = weighted_cross_entropy(y_pred, y_true, edge_map) + (-dice_coef_mean_(y_pred, y_true))
    return loss

# def weighted_cross_entropy_plus_dice_multihead(y_pred_liver, y_true_liver, y_pred_lung, y_true_lung, y_pred_kidney, y_true_kidney, num_classes=3):
#     loss_1 = weighted_cross_entropy(y_pred_liver, y_true_liver, num_classes) + (-dice_coef_weighted_(y_pred_liver, y_true_liver))
#     loss_2 = weighted_cross_entropy(y_pred_lung, y_true_lung, num_classes) + (-dice_coef_weighted_(y_pred_lung, y_true_lung))
#     loss_3 = weighted_cross_entropy(y_pred_kidney, y_true_kidney, num_classes) + (-dice_coef_weighted_(y_pred_kidney, y_true_lung))
#     loss = loss_1 + loss_2 + loss_3
#     return loss

def dice_coef_sum(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


# %%WEIGHTED-DICE-COEFFICIENT
def dice_coef_mean_(y_pred, y_true):

    H, W, num_classes = y_pred.get_shape().as_list()[1:]
    intersection, denominator = 0, 0
    # # print(len(class_weights_dict))
    # norm_weight_list = [float(i) / sum(class_weights_list) for i in class_weights_list]
    # norm_weight_list = [float(i) * 40 for i in norm_weight_list]
    # get median value
    weight_list_sorted = sorted(fre)
    median_value = statistics.median(weight_list_sorted)
    # class weights
    f_dice = [median_value / v for v in fre]
    f_dice[0] = f_dice[0] * 150
    for i in range(num_classes):
        intersection += f_dice[i] * tf.reduce_sum(y_pred[:, :, :, i] * y_true[:, :, :, i])
        denominator += f_dice[i] * (tf.reduce_sum(y_pred[:, :, :, i] + y_true[:, :, :, i]))
    dice_coef_weighted = ((2. * intersection + smooth) / (denominator + smooth))
    return tf.reduce_mean(dice_coef_weighted)


def loss_new(logits, trn_labels, num_classes):
    logits = tf.reshape(logits, [-1, num_classes])
    trn_labels = tf.reshape(trn_labels, [-1, num_classes])
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trn_labels)
    loss = tf.reduce_mean(cross_entropy, name='x_ent_mean')
    return loss


def make_train_op(logits, trn_labels_batch, learning, momentum, update_ops, edge_map):
    # make_train_op(pred1, y, learning, weight_decay, update_ops, edge_map)
    global_step = tf.train.get_or_create_global_step()
    loss = weighted_cross_entropy_plus_dice(logits, trn_labels_batch, edge_map)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.AdamOptimizer(learning).minimize(loss, global_step=global_step)
        # train_op = tf.contrib.opt.AdamWOptimizer(weight_decay).minimize(loss, global_step=global_step)
        train_op = tf.train.MomentumOptimizer(learning, momentum).minimize(loss, global_step=global_step)
    return train_op
