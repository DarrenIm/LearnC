#!/usr/bin/env python
# coding: utf-8

# In[31]:


import nibabel as nib
import numpy as np
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt


# In[32]:


# 将一个list中的mgz文件全部转换成一个tfrecord文件
def conver_to_tfrecord(file_path, output_path, purpose = "train"):
    # 得到image、label、list的filepath
    image_path = join(file_path, "image")
    label_path = join(file_path, "label")
    list_path = join(file_path, purpose+".list")
    # 取得文件名列表
    f = open(list_path)
    file_names = f.readlines()
    # 取得文件数量
    num_file = len(file_names)
    # 输出tfrecord文件的文件名
    output_name = purpose+".tfrecord"
    output = join(output_path, output_name)
    #创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(output)
    print("Writing " + output_name)
    # 观察进度
    flag = 0

    # 遍历文件名列表
    for file_name in file_names:
        # 先做string裁剪处理
        file_name = file_name.strip('\n')
        # 读取一个case的原文件
        mgz_path_original = join(image_path, file_name)
        nparray_original = read_mgz(mgz_path_original)
        # 读取一个case的label
        mgz_path_label = join(label_path, file_name)
        nparray_label = read_mgz(mgz_path_label)
        # map过程可能出错，用try...except语句捕获出错文件名，然后继续写入tfrecord
        try:
            nparray_label = map_nparray_label(nparray_label) #读取的nparray_label经过映射处理
        except Exception:
            print(file_name)
        # 将一个case及其label插入到tfrecords中
        write_into_tfrecord(nparray_original, nparray_label, writer)        
        #观察进度
        flag += 1
        if flag % 5 == 0:
            print(flag)
        if flag % 5 == 0:
            break
    
    f.close()
    writer.close()
    print("Writing End")


# In[33]:


# 给定mgz文件的路径，读取文件，返回一个mgz的numpy_array
def read_mgz(mgz_path):
    data = nib.load(mgz_path)
    np_array = data.get_data()
    # 增加一个维度
    np_array = np_array[...,np.newaxis]
    return np_array


# In[34]:


# 将nparray_label按map映射
def map_nparray_label(nparray_label):
    # map文件路径
    map_path = "/data/brain_segmentation/label_index_map.csv"
    # 根据map文件创建label_map
    label_map = create_map(map_path)
    # 将nparray_label转换为list
    list_label = nparray_label.reshape(-1).tolist()
    # 根据label_map修改list_label的值
    list_label = list(map(lambda x: label_map[x], list_label))
    # 将list_label转回nparray_label
    nparray_label = np.array(list_label).reshape(nparray_label.shape)
    return nparray_label


# In[35]:


# 建立label的映射字典
def create_map(map_path):
    # 建立空map
    label_map = {}
    # 打开文件
    f = open(map_path)
    lines = f.readlines()
    # 删除第一行（标签）
    del lines[0]
    # 得到每一条映射信息，添加到map中
    for line in lines:
        # 先裁剪，把"/n"去掉
        line = line.strip("\n")
        # 分割得到list
        informations = line.split(",")
        # 根据list建立map
        label_map[int(informations[2])] = int(informations[1])
    return label_map


# In[36]:


# 将一个case及其label插入到tfrecords中
def write_into_tfrecord(img2d_list, label2d_list, writer):
    # img2d的个数
    num = img2d_list.shape[0]
    for i in range(num):
        # 将每个2d切片转换成字符串
        img_raw = img2d_list[i].astype(np.uint8).tostring()
        # 将每个label转换成字符串
        label_raw = label2d_list[i].astype(np.uint8).tostring()
        #将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))  
        }))
        #将example写入TFRecord文件
        writer.write(example.SerializeToString())


# In[37]:


# 描述features的字典
dataset_features = {
    'image': tf.FixedLenFeature([],tf.string),
    'label': tf.FixedLenFeature([],tf.string)
}
    
# 对每个二进制的example,将其转换为tensor
def _parse_dataset(example_proto):
    return tf.parse_single_example(example_proto, dataset_features)

# 从tfrecord文件中读取数据
def read_from_tfrecord(dir_path,purpose,epochs,batch_size):
    # 根据purpose建立要解析的文件路径
    file_path = join(dir_path, purpose+".tfrecord")
    # 解析tfrecord中的所有二进制的examples
    raw_dataset = tf.data.TFRecordDataset(file_path)
    # 解析二进制的examples为tensor
    parsed_data = raw_dataset.map(_parse_dataset) 
    # 设定parsed_data的epochs,batch_size
    parsed_data = parsed_data.repeat(epochs).shuffle(5000).batch(batch_size)
    # 取出数据
    iterator = parsed_data.make_one_shot_iterator()
    features = iterator.get_next()
    return features


# In[39]:


dir_path = "/data/tfrecords"
purpose = "train"
epochs = 10
batch_size = 30
features = read_from_tfrecord(dir_path,purpose,epochs,batch_size)


# In[40]:


sess = tf.Session()
first_batch = sess.run(features)
second_batch = sess.run(features)


# In[69]:


a = 3
img1 = np.frombuffer(first_batch["image"][a], dtype=np.uint8)
label1 = np.frombuffer(first_batch["label"][a], dtype=np.uint8)
img2 = np.frombuffer(second_batch["image"][a], dtype=np.uint8)
label2 = np.frombuffer(second_batch["label"][a], dtype=np.uint8)
img1 = img1.reshape((256,256))
label1 = label1.reshape((256,256))
img2 = img2.reshape((256,256))
label2 = label2.reshape((256,256))


# In[70]:


plt.figure()
plt.subplot(221)
plt.imshow(img1)
plt.subplot(223)
plt.imshow(label1)
plt.subplot(222)
plt.imshow(img2)
plt.subplot(224)
plt.imshow(label2)
plt.show()


# In[71]:


sess.close()


# In[ ]:




