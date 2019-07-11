import os
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model import BrainSegmentation

def dice_score(pred, label, num_classes=40):
    eps = 1e-5
    dice = []
    for c in range(num_classes):
        c_pred = np.zeros_like(pred).astype(np.int)
        c_label = np.zeros_like(label).astype(np.int)
        c_pred[pred == c] = 1
        c_label[label == c] = 1
        numerator = np.sum(2 * c_pred * c_label)
        denominator = np.sum(c_pred + c_label) + eps
        dice.append(numerator / denominator)
    return dice

def get_volumn(pred, num_classes=40):
    vol = []
    for c in range(num_classes):
        c_vol = np.sum(pred == c) * 8
        vol.append(str(c_vol))
    return vol

if __name__ == "__main__":
    model_path = './results/model_3d_128/epoch_17_val_loss_-0.83740'
    resample_label = True
    mat = np.diag([2., 2., 2.])

    model = BrainSegmentation(num_classes=40)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    mean_dice_r = []
    mean_dice_l = []

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
    with open('/data/brain_segmentation/test_cadni.csv') as f, open('./vol.csv', 'w') as out:
        for i, line in enumerate(f):
            if i == 0:
                out.write(','.join(['name'] + brain_name))
                out.write('\n')
                continue
            line = line.strip().split(',')
            img = nib.load(line[0])
            label = nib.load(line[1])

            vec = img.affine.dot(np.array(img.shape + (2,)) // 2)[:3] + np.array([-128, -128, -128])
            new_affine = nib.affines.from_matvec(mat, vec)

            if resample_label:
                label = resample_from_to(label, [[128, 128, 128], new_affine], order=0).get_fdata()
            else:
                label = label.get_fdata()
            
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
            resample_img = resample_from_to(scale_img, [[128, 128, 128], new_affine]).get_fdata()
            resample_img = torch.from_numpy(resample_img[np.newaxis, np.newaxis, ...]).float()

            pred = torch.argmax(F.softmax(model(resample_img.cuda()), 1), dim=1).cpu().numpy()
            pred = np.squeeze(pred)

            if not resample_label:
                pred = nib.nifti1.Nifti1Image(pred.astype(np.uint8), affine=new_affine)
                pred = resample_from_to(pred, [img.shape, img.affine], order=0)
                pred = pred.get_fdata()

            dice = dice_score(pred, new_label)
            vol = get_volumn(pred)
            out.write(','.join([line[0].split('/')[-1]] + vol))
            out.write('\n')
            res_dice.append(dice)

            # if i % 20 == 0:
            #     mean_dice = np.mean(np.array(res_dice), axis=0)
            #     print('Mean dice', np.mean(mean_dice))
            #     for n, d in zip(brain_name, mean_dice):
            #         print(n, d)
            #     print("====" * 20)
    mean_dice = np.mean(np.array(res_dice), axis=0)
    print('Mean dice', np.mean(mean_dice))
    for n, d in zip(brain_name, mean_dice):
        print(n, d)