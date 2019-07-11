#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
import scipy
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import ndimage
import random
import operator
import re
from PIL import Image
from itertools import chain, islice
import cv2

def ichunked(seq, chunksize):
    """Yields items from an iterator in iterable chunks."""
    it = iter(seq)
    while True:
        try:
            yield chain([next(it)], islice(it, chunksize - 1))
        except StopIteration:
            return


def chunked(seq, chunksize):
    """Yields items from an iterator in list chunks."""
    for chunk in ichunked(seq, chunksize):
        yield list(chunk)


def get_number(x):
    regex = re.compile(r'\d+')
    return [int(x) for x in regex.findall(x)]


def getdimentiondiff(size):
    # get pad widths for each axis
    row = abs(size[0] - 256)
    column = abs(size[1] - 256)
    return row, column


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    # listOfFile = listOfFile.sort()
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def get_picklenames(dirName):
    filelist = os.listdir(dirName)
    # print(filelist)
    filelist.sort(key=lambda x: int(x[:-7]))
    filepaths = [os.path.join(dirName, x) for x in filelist]
    return filepaths


def get_list_zero(data3D, dep):
    idx = []
    for i in range(dep):
        if np.all(data3D[i] == 0):
            idx.append(i)
        else:
            pass
    return idx

def loadfilename(filename, zeros, axis="sagittal", label=False):
    # load nii
    # return padded 3D nparray and allzero slice index
    if label==True:
        datanii = nib.load(filename)
        data3D = np.array(datanii.dataobj)
        aff = datanii.affine[0:3, 0:3]
        index = aff.argmin(axis=1)
        Slicelist = []
        if axis == "sagittal":
            # index and value of min (0,1,2) or (2,1,0)
            k, value = min(enumerate(index), key=operator.itemgetter(1))
            if k == 2:
                row, column = getdimentiondiff(data3D[..., 0].shape)
                dep = data3D.shape[k]
                zerolist = get_list_zero(data3D, dep)
                for i in range(dep):# [256, 256, 166], dep = 166
                    # pad at bottom right
                    if i in zerolist:
                        pass
                    else:
                        data256 = np.pad(data3D[...,i], ((row,0),(column,0)), 'constant', constant_values=0)
                        Slicelist.append(data256)
                padded3D = np.dstack(Slicelist)
                return padded3D, zerolist # return [256, 256, dep]
            else:
                row, column = getdimentiondiff(data3D[0, ...].shape)
                dep = data3D.shape[k]
                zerolist = get_list_zero(data3D, dep)
                data256 = np.zeros(shape=(dep, 256, 256), dtype=np.float32)

                for i in range(dep):# [166, 256, 256], dep = 166
                    # pad center at bottom right
                    if i in zerolist:
                        pass
                    else:
                        data256[i] = np.pad(data3D[i,...],((row,0),(column,0)), 'constant', constant_values=0)
                        Slicelist.append(data256)
                padded3D = np.dstack(Slicelist)
                return padded3D, zerolist # return [256, 256, dep]
    else:
        datanii = nib.load(filename)
        data3D = np.array(datanii.dataobj)
        aff = datanii.affine[0:3, 0:3]
        index = aff.argmin(axis=1)
        Slicelist = []
        if axis == "sagittal":
            # index and value of min (0,1,2) or (2,1,0)
            k, value = min(enumerate(index), key=operator.itemgetter(1))
            if k == 2:
                row, column = getdimentiondiff(data3D[..., 0].shape)
                dep = data3D.shape[k]
                for i in range(dep):  # [256, 256, 166], dep = 166
                    if i in zeros:
                        pass
                    else:
                        # pad at bottom right
                        data256 = np.pad(data3D[..., i], ((row, 0), (column, 0)), 'constant', constant_values=0)
                        Slicelist.append(data256)
                padded3D = np.dstack(Slicelist)
                return padded3D  # return [256, 256, dep]
            else:
                row, column = getdimentiondiff(data3D[0, ...].shape)
                dep = data3D.shape[k]
                for i in range(dep):  # [166, 256, 256], dep = 166
                    # pad center at bottom right
                    if i in zeros:
                        pass
                    else:
                        data256 = np.pad(data3D[i, ...], ((row, 0), (column, 0)), 'constant', constant_values=0)
                        Slicelist.append(data256)
                padded3D = np.dstack(Slicelist)
                return padded3D


def create_pickle(data, index, path, label=False):
# all pickle from one MRI
    j = 0
    idx = index[-1]
    if label == False:
        for i in range(data.shape[-1]):
            image_patch = data[:,:,i]
            with open(path + ('/No.%d_%d.pickle' % (idx, j)), 'wb') as f1:
                pickle.dump(image_patch, f1)
            j += 1
    else:
        for i in range(data.shape[-1]):
            label_patch = data[:,:,i]
            with open(path + ('/No.%d_%d.pickle' % (idx, j)), 'wb') as f2:
                pickle.dump(label_patch, f2)
            j += 1

def read_all_pickles(imglist, labellist, imgpath, labelpath, axis="sagittal", ):
    for n, x_train_file in enumerate(imglist):
        img_numbers = get_number(x_train_file)
        label_numbers = get_number(labellist[n])
        if img_numbers != label_numbers:
            raise ValueError("img and label pickles matching error")
        else:
            # create pickles with non-zero brain label from one input 3D nii
            I = []
            label3D, zeroslice = loadfilename(labellist[n], I, axis="sagittal", label=True)
            img3D = loadfilename(x_train_file, zeroslice, axis="sagittal")
            create_pickle(img3D, img_numbers, imgpath)
            create_pickle(label3D, label_numbers, labelpath, label=True)
    x_train_picklenames = getListOfFiles(imgpath)
    y_train_picklenames = getListOfFiles(labelpath)
    return x_train_picklenames, y_train_picklenames


def data_augmentation(batch_data, flip_lr=True, rotate=False, noise=True):
    if flip_lr:
        batch_data = _random_flip_leftright(batch_data)
    if rotate:
        batch_data = _random_rotation(batch_data, 10)
    if noise:
        batch_data = _random_blur(batch_data, 2)
    return batch_data


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Random angle
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle,
                                                          reshape=False)
    return batch


def _random_blur(batch, sigma_max):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            # Random sigma
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def split_dataset(labels, dataset="liver_spleen"):
    if dataset == "liver_spleen":
        label_liver_spleen = np.copy(np.array(labels))
        for label in label_liver_spleen:
            # indexed array
            for i in range(4, 12):
                label[0, :, :][label[0, :, :] == i] = 1
        return label_liver_spleen[:, 0, :, :]


    if dataset == "rlung_llung":
        label_rlung_llung = np.copy(np.array(labels))
        for label in label_rlung_llung:
            # indexed array
            for i in range(8, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(2, 6):
                label[0, :, :][label[0, :, :] == j] = 1
        return label_rlung_llung[:, 0, :, :]

    if dataset == "rkidney_lkidney":
        label_rkidney_lkidney = np.copy(np.array(labels))
        for label in label_rkidney_lkidney:
            # indexed array
            for i in range(10, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(2, 8):
                label[0, :, :][label[0, :, :] == j] = 1
        return label_rkidney_lkidney[:, 0, :, :]


    if dataset == "liver_kidney_lung":
        liver_kidney_lung = np.copy(np.array(labels))
        for label in liver_kidney_lung:
            for i in range(10, 12):
                label[0, :, :][label[0, :, :] == i] = 1
            for j in range(4, 6):
                label[0, :, :][label[0, :, :] == j] = 1
        return liver_kidney_lung[:, 0, :, :]


def one_hot_encode(label, num_classes):
    # if (num_classes == 3):
    #     label[label == label.max()] = 3
    #     label[label == 6] = 2
    #     label[label == 8] = 2
    #     label_ohe = (np.arange(num_classes) == np.array(label)[..., None]).astype(int)
    # else:
    label_ohe = (np.arange(num_classes) == np.array(label)[..., None]).astype(int)
    return label_ohe


def remove_back_pixels(train_data, train_labels):
    """
    train_labels: without one hot encoding
    """
    idx = []
    for i in range(0, train_labels.shape[0]):
        if (len(np.unique(train_labels[i])) != 1):
            idx.append(i)
    train_data = np.array([train_data[i] for i in idx])
    train_labels = np.array([train_labels[i] for i in idx])
    return train_data, train_labels


def centeredCrop(img, new_height, new_width):
    batch = []
    width = np.size(img[0], 1)
    height = np.size(img[0], 0)

    left = int(np.ceil((width - new_width) / 2.))
    top = int(np.ceil((height - new_height) / 2.))
    right = int(np.floor((width + new_width) / 2.))
    bottom = int(np.floor((height + new_height) / 2.))
    for i in range(len(img)):
        batch.append(img[i][top:bottom, left:right])
    return np.array(batch)


def normalize_data(slice):
    min = np.amin(slice)
    max = np.amax(slice)
    if min == max == 0 or min == max:
        return slice
    else:
        return ((slice - min) / (max - min))

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128,  64, 128), (244,  35, 232),
                 (70,  70,  70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170,  30), (220, 220,   0),
                 (107, 142,  35), (152, 251, 152), (70, 130, 180), (220,  20,  60), (255, 0, 0), (0, 0, 142), (0, 60, 100),
                 (128,  64, 128), (244,  35, 232), (70,  70,  70), (102, 102, 156)]




def data_generator(batch_size, imglist, map_dict, num_classes):

    X = np.zeros(shape=(batch_size, 256, 256), dtype=np.float32)
    Y = np.zeros(shape=(batch_size, 256, 256), dtype=np.float32)
    edge_np = np.zeros(shape=(batch_size, 256, 256), dtype=np.int32)

    for n, filename in enumerate(imglist):
        filename = filename.strip()
        path = os.path.join('/data/brain_segmentation/frontal/image', filename)
        img_np = Image.open(path)
        img_np = np.asarray(img_np)
        X[n] = normalize_data(img_np)
    X = np.expand_dims(X, -1)

    for m, labelname in enumerate(imglist):
        labelname = labelname.strip()
        path = os.path.join('/data/brain_segmentation/frontal/label', labelname)
        label_np = Image.open(path)
        label_np = np.asarray(label_np)
        # map pixel value to [0, 39]
        label_np_mapped = np.vectorize(map_dict.get)(label_np)
        Y[m] = label_np_mapped
        assert not np.any(np.isnan(Y[m]))
        # create edge map by canny filter
        edge_np[m] = cv2.Canny(np.uint8(label_np_mapped), 1, 2)
    label_ohe = one_hot_encode(Y, num_classes)

    return X, label_ohe, edge_np




