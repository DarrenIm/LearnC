#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from PIL import Image
import pickle
import time
from tqdm import tqdm


def label_distribution_calculate():
    valid_dir = '/data/tfrecords/sagittal/nonzeros_valid.list'
    train_dir = '/data/tfrecords/sagittal/nonzeros_train.list'
    dict_list = []

    with open(train_dir) as file:
        train_dir = list(file)

    print("----counting values----")

    for filename in tqdm(train_dir):
        filename = filename.strip()
        path = os.path.join('/data/brain_segmentation/sagittal/label', filename)
        img_np = Image.open(path)
        img_np = np.asarray(img_np)
        # if np.all(img_np == 0):
        #     pass
        # else:
        unique, counts = np.unique(img_np, return_counts=True)
        value_dict = dict(zip(unique, counts))
        dict_list.append(value_dict)

    # dict_list = [{'a':1, 'b':2, 'c':3},
    #          {'a':1, 'd':2, 'c':5},
    #          {'e':57, 'c':3} ]
    # ----update dicts by summing values----
    c = collections.Counter()
    for d in dict_list:
        c.update(d)
    sum_list = dict(c)

    # ----calculating distribution----
    print("v / (number_of_png * 256 * 256)")
    total_number = (len(dict_list) * 256 * 256)
    sum_list = {k: v / total_number for k, v in sum_list.items()}
    print(sum_list)
    with open('label_distribution_nonzero.pickle', 'wb') as f:
        pickle.dump(sum_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    label_distribution_calculate()


