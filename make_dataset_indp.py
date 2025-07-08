import os
from cv2 import NORM_MINMAX
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import TrainParam

root_path = "../DATA/dataset_26c_50w_100ws_30st/ZCL"

datas = np.load(os.path.join(root_path, "datas.npy"), allow_pickle=True).tolist()
labels = np.load(os.path.join(root_path, "labels.npy"), allow_pickle=True).tolist()
# userids = np.load(os.path.join(root_path, "userids.npy"), allow_pickle=True)

save_path = "../DATA/dataset_independence/ZCL"
if os.path.exists(save_path) is False:
    os.mkdir(save_path)

# print(labels[0].shape)
# exit(0)

# data padding
nMaxLength = 0
for data in datas:
    nMaxLength = max(data.shape[0], nMaxLength)

new_datas = []
datas_length = []
for data in datas:
    nPadNum = nMaxLength - data.shape[0]
    datas_length.append(data.shape[0])
    data = np.pad(data, ((0, nPadNum),(0,0),(0,0),(0,0)), 'constant')
    new_datas.append(data)

new_datas = np.array(new_datas)
datas_length = np.array(datas_length)

# target padding
nMaxLength = 0
for label in labels:
    nMaxLength = max(label.shape[0], nMaxLength)

new_labels = []
labels_length = []
for label in labels:
    nPadNum = nMaxLength - label.shape[0]
    labels_length.append(label.shape[0])
    label = np.pad(label, (0, nPadNum), mode='constant', constant_values=27)
    new_labels.append(label)

new_labels = np.array(new_labels)
labels_length = np.array(labels_length)
print(new_labels.shape)
print(labels_length.shape)


# int 标签转 str 标签
# def int2char(label) -> str:
#     result = ""
#     for l in label:
#         if l >= len(TrainParam.CHARS):       # 当前值为padding的
#             continue
#         else:
#             result += TrainParam.CHARS[l]  # 当前值是正常值
    
#     return result
    
# int_labels = labels.tolist()
# str_labels = []
# for i_label in int_labels:
#     s_label = int2char(i_label)
#     str_labels.append(s_label)

# # 生成标签set
# label_set = np.array(list(set(str_labels)))

# # 取40个作为训练 10个作为测试
# index = np.arange(0, len(label_set))
# np.random.shuffle(index)
# train_label = label_set[index[:40]]
# test_label = label_set[index[40:]]

# # 生成 train 和 test 样本的下标集
# train_indexs = []
# test_indexs = []
# for i in range(len(str_labels)):
#     if str_labels[i] in train_label:
#         train_indexs.append(i)
#     elif str_labels[i] in test_label:
#         test_indexs.append(i)
 
# train_indexs = np.array(train_indexs)
# test_indexs = np.array(test_indexs)

# 保存数据
np.save(os.path.join(save_path, "new_datas.npy"), new_datas)
np.save(os.path.join(save_path, "datas_length.npy"), datas_length)
np.save(os.path.join(save_path, "new_labels.npy"), new_labels)
np.save(os.path.join(save_path, "labels_length.npy"), labels_length)
# np.save(os.path.join(save_path, "userids.npy"), userids)
# np.save(os.path.join(save_path, "train_dataset_indexs.npy"), train_indexs)
# np.save(os.path.join(save_path, "test_dataset_indexs.npy"), test_indexs)

print("finish!")