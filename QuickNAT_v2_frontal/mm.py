a = [0.99575769,0.92806156,0.86727156, 0.88572313, 0.76512821, 0.86674454,
 0.88786417, 0.8241048,  0.78849626, 0.85923609, 0.84120172, 0.79568442,
 0.71875,    0.88197697, 0.85375218, 0.83193015, 0.65101038, 0.74987107,
 0.80630117, 0.47863248, 0.91937262, 0.8728686,  0.85645064, 0.79906542,
 0.84640602, 0.8944045 , 0.91269841, 0.88918416, 0.91487271, 0.80452489,
 0.86537411, 0.80763116, 0.80974125, 0.84103318, 0.63874346, 0.53660797,
 0.56384065, 0.5476025,  0.35855856, 0.6031614]

# print(sum(a)/len(a))

b = [4.92768480e-01,0.00000000e+00, 5.88235294e-05, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.54715078e-02,
 1.68855535e-02, 1.78537957e-01, 0.00000000e+00, 0.00000000e+00,
 5.39224527e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 2.72221960e-03, 2.57984613e-03, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 3.88500388e-04, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 1.27039969e-03, 0.00000000e+00, 0.00000000e+00,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

# print([x * 2 for x in b])

import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image
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
    print(p)
    path = os.path.join(p, 'saved_models')
    for data, label in generator:
        assert not np.any(np.isnan(data))
        assert not np.any(np.isnan(label))

        for i in range(label.shape[-1]):
            # print(label.shape)
            unique, counts = np.unique(label[:,:,i], return_counts=True)
            value_dict = dict(zip(unique, counts))
            print(value_dict)
            # im = Image.fromarray(label[:,:,i])
            #
            # im = im.convert("L")
            # im.save(p + "/pngs/label/label_file_%d.png" % i, format='PNG')
            # png_p = p + "/pngs/png_file_55.png"
            # img = Image.open(png_p)
            # print(np.equal(img, data[:,:,55]))
        break
        # new_image = nib.Nifti1Image(label, affine=np.eye(4))
        # nib.save(new_image, 'test_label.nii')
        # break
if __name__ == '__main__':
    predict()