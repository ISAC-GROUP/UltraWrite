# =================================
#   创建训练集和测试集
# =================================


import numpy as np
from utils import util
import TrainParam
# from Process import SignalProcess, Display
import os
import matplotlib.pyplot as plt
import TrainParam
import math

# 将String类别的标签转化为int标签
def char2label(label):
    result = []
    for ch in label:
        result.append(
            TrainParam.CHAR2INT[ch]
        )

    result = np.array(result)
    return result


ROOT_PATH = "../DATA/ALL/final_real_word/ZPZ_WORD_v3"
SAVE_PATH = "../DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2"

if os.path.exists(SAVE_PATH) is False:
    os.mkdir(SAVE_PATH)

# 原始数据，每个单词保存为一个npy文件，类别为文件名的第一个字段，用户名为数据对应的父目录名
# words, word_labels, word_userids = util.GetData(root_path=TrainParam.DATA_PATH)
words, word_labels, word_ids = util.GetData(root_path=ROOT_PATH)
datas = []
labels = []
ids = []

# for i in range(len(word_labels)):
#     _, _, word = SignalProcess.preprocess(words[i], nperseg=TrainParam.NPERSEG, noverlap=TrainParam.NOVERLAP, FS=TrainParam.FS, log_flag=True, flip_flag=False)
#     np.save(os.path.join(SAVE_PATH, userids[i]), word)

# exit(0)


# exit(0)
# count = 0
# 对每个单词进行预处理，并且单词内进行切分成多个时间片
# for word in words:
for i in range(len(word_labels)):
    # _, _, word = SignalProcess.preprocess(words[i], nperseg=TrainParam.NPERSEG, noverlap=TrainParam.NOVERLAP, FS=TrainParam.FS, log_flag=True, flip_flag=True)
    word = np.flip(words[i],axis=0)
    # 记录每个词切分后的序列
    
    flag = False
    if (((word.shape[1] - TrainParam.WINDOW_SIZE) / TrainParam.STRIDE) + 1 < len(word_labels[i])):
        pad_size = ((len(word_labels[i]) - 1) * TrainParam.STRIDE + TrainParam.WINDOW_SIZE - word.shape[1])
        pad_size = math.ceil(pad_size / 2)
        word = np.pad(word, ((0,0),(pad_size, pad_size)), mode="constant")
        flag = True
    
    t = word.shape[1]
    data = []
    index = 0

    while (index + TrainParam.WINDOW_SIZE) <= t:
        d = word[:, index:index+TrainParam.WINDOW_SIZE]
        # if d.shape[1] < TrainParam.WINDOW_SIZE:
        #     d = np.pad(d, ((0,0), (0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
        d = d.reshape(1, d.shape[0], d.shape[1])
        data.append(d)
        
        index += TrainParam.STRIDE


    # if(index + (TrainParam.WINDOW_SIZE / 2) < t):
    # if ((t - index - 1) >= (TrainParam.WINDOW_SIZE - TrainParam.STRIDE)):
    if(flag is False):
        d = word[:, index:]
        # if(d.shape[1] < TrainParam.WINDOW_SIZE):
        #     d = np.pad(d, ((0,0),(0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
        #     d = d.reshape(1, d.shape[0], d.shape[1])
        #     data.append(d)
        d = np.pad(d, ((0,0),(0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
        d = d.reshape(1, d.shape[0], d.shape[1])
        data.append(d)


    data = np.array(data)
    label = char2label(word_labels[i])
    if label.shape[0] <= data.shape[0]:
        datas.append(data)
        labels.append(label)
        # uids.append(word_userids[i])



# exit(0)
# 根据比例切分为训练集和测试集
# train_data_count = int(len(labels) * TrainParam.TRAIN_TEST_RATE)

# indexs = np.arange(0, len(labels))
# np.random.shuffle(indexs)                                       # 打散indexs

# 根据比例获取前train_data_count作为训练集，其余的作为测试集
# train_indexs = indexs[:train_data_count]
# test_indexs = indexs[train_data_count:-1]

datas = np.array(datas, dtype=np.ndarray)
labels = np.array(labels, dtype=np.ndarray)

for id in word_ids:
    ids.append(int(id))

ids = np.array(ids, dtype=np.int32)

# uids = np.array(uids, dtype=np.ndarray)
#
# if os.path.exists(TrainParam.DATASET_PATH) is False:
#     os.mkdir(TrainParam.DATASET_PATH)

np.save(os.path.join(SAVE_PATH, "datas.npy"), datas)
np.save(os.path.join(SAVE_PATH, "labels.npy"), labels)
np.save(os.path.join(SAVE_PATH, "ids.npy"), ids)
# np.save(os.path.join(TrainParam.DATASET_PATH, "datas.npy"), datas)
# np.save(os.path.join(TrainParam.DATASET_PATH, "labels.npy"), labels)
# np.save(os.path.join(TrainParam.DATASET_PATH, "userids.npy"), uids)
# np.save(os.path.join(TrainParam.DATASET_PATH, "train_dataset_indexs.npy"), train_indexs)
# np.save(os.path.join(TrainParam.DATASET_PATH, "test_dataset_indexs.npy"), test_indexs)

print("make dataset successful!")

