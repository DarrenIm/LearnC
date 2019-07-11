import os
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

def resampled_nii_generator():
    '''
    :return: np array with shape [256, 256, 256]
    '''
    mat = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    label_map = dict()
    brain_name = []
    with open('/data/brain_segmentation/label_index_map_new.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split(',')
            brain_name.append(line[0])
            label_map[int(line[2])] = int(line[1])

    res_dice = []
    with open('/data/brain_segmentation/test_cadni.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                # out.write(','.join(['name'] + brain_name))
                # out.write('\n')
                continue
            line = line.strip().split(',')
            img = nib.load(line[0])
            label = nib.load(line[1])

            vec = img.affine.dot(np.array(img.shape + (2,)) // 2)[:3] + np.array([128, -128, 128])
            new_affine = nib.affines.from_matvec(mat, vec)


            label = resample_from_to(label, [[256, 256, 256], new_affine], order=0).get_fdata()
            # label = label.get_fdata()

            new_label = np.zeros_like(label)

            for k, v in label_map.items():
                new_label[label == k] = v

            img_data = img.get_fdata()
            hist = np.histogram(img_data, bins=1000)
            count = 0
            for b, nb in enumerate(hist[0][::-1]):
                count += nb
                if count > 0.001 * img.shape[0] * img.shape[1] * img.shape[2]:
                    break
            img_min = hist[1][0]
            img_max = hist[1][::-1][b]

            if img_max < img_min:
                print('Error in file %s.' % line[0])
                continue
            scale = 255. / (img_max - img_min)
            img_data = (img_data - img_min) * scale
            img_data[img_data > 255] = 255
            img_data[img_data < 0] = 0
            img_data[img.get_fdata() == 0] = 0
            scale_img = nib.nifti1.Nifti1Image(img_data.astype(np.uint8), affine=img.affine)
            resample_img = resample_from_to(scale_img, [[256, 256, 256], new_affine], order=1).get_fdata()
            yield resample_img, new_label, new_affine, [line[0].split('/')[-1]]