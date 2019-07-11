import csv
import pickle
import numpy as np
# list_value = [19, 35, 37, 38, 39, 40, 41, 42, 43, 44]
# with open('label_distribution.pickle', 'rb') as f1:
#     weight_dict = pickle.load(f1)
#     print(weight_dict)
#     new_dir = {}
#     value = 0
#     for k, v in weight_dict.items():
#         if k in list_value:
#             value += v
#     for i in range(50):
#         if i in list_value:
#             new_dir[i] = 0
#         elif i == 0:
#             new_dir[i] = weight_dict[0] + value
#         else:
#             new_dir[i] = weight_dict[i]
#     print(new_dir)
#     for i in range(50):
#         if i in list_value:
#             del new_dir[i]
#         else:
#             pass
#     print(new_dir)
#     list_40 = list(range(40))
#     dir_0 = dict.fromkeys(list_40, 0)
#     for k1, k2 in zip(new_dir,dir_0):
#         dir_0[k2] = new_dir[k1]
#     print(dir_0)
#     print(sum(dir_0.values()))
#     with open('label_distribution_new.pickle', 'wb') as f:
#         pickle.dump(dir_0, f, pickle.HIGHEST_PROTOCOL)
#
# with open('label_index_map.csv', mode='r') as infile:
#     reader = csv.reader(infile)
#     for rows in reader:
#         olddict = {rows[0]: int(rows[1]) for rows in reader}
#     print(olddict)
#
# with open('label_index_map_new.csv', mode='r') as outfile:
#     reader_new = csv.reader(outfile)
#     for rows in reader_new:
#         newdict = {rows[0]: int(rows[1]) for rows in reader_new}
#     print(newdict)
# map_dict = {}
# for k1, v1 in olddict.items():
#     if k1 in newdict:
#         map_dict[olddict[k1]] = newdict[k1]
#     else:
#         map_dict[olddict[k1]] = 0
#
# print(map_dict)
# with open('map_dictionary.pickle', 'wb') as f:
#     pickle.dump(map_dict, f, pickle.HIGHEST_PROTOCOL)
#
# a = np.array([[1, 2, 3],
#                   [3, 2, 49]])
# b = np.vectorize(map_dict.get)(a)
# print(b)
# with open('./saved_pickles/label_distribution_new.pickle', 'rb') as f:
#     distribution_dict = pickle.load(f)
# class_weights_dict = {k: 1 / v for k, v in distribution_dict.items()}
#     # sorted_weights = collections.OrderedDict(sorted(class_weights_dict.items()))
# class_weights_list = []
# for k, v in class_weights_dict.items():
#     class_weights_list.append(v)
#     # print(len(class_weights_dict))
# norm_weight_list = [float(i) / sum(class_weights_list) for i in class_weights_list]
# print(sum(norm_weight_list))
# norm_weight_list = [float(i) * 40 for i in norm_weight_list]
# print(sum(norm_weight_list))
#
# raw = [0.07, 0.14, 0.07]
# min_r = min(raw)
# max_r = max(raw)
# n_r = list()
# for value in raw:
#     nor = (value - min_r)/ max_r - min_r
#     n_r.append(nor)
# print(sum(n_r))
# for i in range(1,10):
#     print(i)

a = 123
print(a%10)