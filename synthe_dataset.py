# ========================
#       随机合成数据
# ========================


from calendar import c
import numpy as np
import os
import random

import TrainParam
import math
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch

def RandomFactor(Low:int = 0, High:int = 25, flag:str='N', mean = 1, sigma=2) -> int:
    '''
    Args
    ----
        flag(str): 表示使用什么分布     ['N', 'U']
    '''
    if flag == 'N':
        nResult = int(random.gauss(mean, sigma))
        if nResult < Low:
            nResult = Low
        elif nResult > High:
            nResult = High
        return nResult

    if flag == 'U':
        nResult = random.randint(Low, High)
        return nResult

def RandomOverlap(Low:float, High:float, size:int) -> int:
    overlap_rate = random.uniform(Low, High)
    overlap_cols = int(overlap_rate * size)
    return overlap_cols

def merge(mat1:np.ndarray, mat2:np.ndarray)-> torch.Tensor:
    mat = torch.zeros_like(mat1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i][j] = max(mat1[i][j], mat2[i][j])
    return mat

_letter2int = {
    "A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6,\
    "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13,\
    "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19,\
    "U":20, "V":21, "W":22, "X":23, "Y":24, "Z":25
}
_int2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
    7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
    14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
    20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
}

def int2char(label):
    charlabel = ""
    for l in label:
        charlabel += _int2letter[l]
    return charlabel

def char2int(label):
    intlabel = []
    for i in range(len(label)):
        intlabel.append(_letter2int[label[i]])
    return intlabel

# ====================================================================
# read source letter signal
# ====================================================================
root_path = "../DATA/letter/ZCL/after_process_v3"

signals = []
for i in range(10):
    signal_path = "letter_" + str(i+1) + "_signal_ts.npy"
    index_path = "letter_" + str(i+1) + "_index_ts.npy"

    signal = np.load(os.path.join(root_path, signal_path))
    index = np.load(os.path.join(root_path, index_path))
    letter_signal = []
    for idx in index:
        letter_signal.append(signal[:, idx[0]:idx[1]])
    signals.append(letter_signal)

reset_actions = {}
reset_name = ["UU","UD", "MU", "MD", "DU", "DD"]
for i in range(6):
    signal_path = reset_name[i] + "_signal_ts.npy"
    index_path = reset_name[i] + "_index_ts.npy"
    reset_action = []
    if os.path.exists(os.path.join(root_path, signal_path)) is True:
        signal = np.load(os.path.join(root_path, signal_path))
        index = np.load(os.path.join(root_path, index_path))
        for idx in index:
            reset_action.append(signal[:, idx[0]:idx[1]])
    reset_actions[reset_name[i]] = reset_action

letters_start = [
    'U', 'U', 'U', 'U', 'U', 'U', 'U',\
    'U', 'U', 'U', 'U', 'U', 'U', 'U',\
    'D', 'U', 'U', 'U', 'U', 'U',\
    'U', 'U', 'U', 'U', 'U', 'U'
]
letters_end = [
    'M', 'D', 'D', 'D', 'M', 'D', 'M',\
    'D', 'D', 'D', 'D', 'D', 'U', 'D', \
    'D', 'M', 'D', 'D', 'D', 'D',\
    'U', 'U', 'D', 'D', 'D', 'M'
]

dictionarys = [
    'AM', 'AND', 'ARE', 'BACK',\
    'BE', 'BUT', 'CAN', 'COME',\
    'DO', 'EIGHT', 'FAMILY', 'FATHER',\
    'FIVE', 'FOUR', 'GIVE', 'GOOD',\
    'HAVE', 'HE', 'HELLO', 'HOW',\
    'IS', 'JOB', 'JUST', 'MAKE',\
    'MAN', 'ME', 'MOTHER', 'NEXT',\
    'NINE', 'ONE', 'PROJECT', 'PUT',\
    'QUIT', 'REQUIRE', 'SAY', 'SEVEN',\
    'SIX', 'SIZE', 'THAT', 'THE', \
    'THEY', 'THINK', 'THREE', 'TWO',\
    'WHAT', 'WHERE', 'WHY', 'WORLD',\
    'YOU', 'ZERO'
]


stretch = T.TimeStretch(n_freq=114)
# stretch_signals = []

factor_down = {'A':68, 'B':80, 'C':25, 'D':60, 'E': 70, 'F':75, 'G':60,\
          'H': 60, 'I': 20,'J': 58, 'K':60, 'L':38, 'M':85, 'N':70,\
          'O':40, 'P':73, 'Q':75, 'R':70, 'S':40, 'T':55,\
          'U':48, 'V':45, 'W':85, 'X':45, 'Y':60, 'Z':78}

factor_up = {'A':85, 'B':85, 'C':35, 'D':75, 'E': 90, 'F':95, 'G':70,\
          'H': 85, 'I': 28,'J': 70, 'K':72, 'L':45, 'M':105, 'N':95,\
          'O':50, 'P':77, 'Q':82, 'R':90, 'S':55, 'T':65,\
          'U':58, 'V':55, 'W':115, 'X':65, 'Y':80, 'Z':88}

# EXP_COUNT = 5

# for i in range(len(signals)): # 10
#     stretch_signals_letter = []
#     for k in range(EXP_COUNT):
#         stretch_signals_letter.append([])
#     for j in range(26):      # 26
#         if():
#             stretch_signals_letter.append(signals[i][j])
#         else:
#             source_signal = torch.from_numpy(np.ascontiguousarray(signals[i][j]))
#             # factor = np.random.uniform(low=0.1, high=(max(0.6, 1-0.1*len(label))), size=len(label))
#             rate = source_signal.shape[1] / factor[_int2letter[j]]

#             warp_signal = torch.abs(stretch(source_signal, rate))
#             stretch_signals_letter.append(warp_signal.numpy())
#     stretch_signals.append(stretch_signals_letter)

reset_factor_down = {'UD':24, 'MU':20, 'MD':20, 'DU':24}
reset_factor_up = {'UD':32, 'MU':28, 'MD':28, 'DU':32}
# reset_factor = {'UD':26, 'MU':20, 'MD':20, 'DU':26}
# stretch_reset_actions = {}
# for (key, vals) in reset_actions.items():
#     if(key == 'UU' or key == 'DD'):
#         stretch_reset_actions[key] = []

#     stretch_reset_action = []
#     for val in vals:
#         source_signal = torch.from_numpy(np.ascontiguousarray(val))
#         rate = source_signal.shape[1] / reset_factor[key]
#         warp_signal = torch.abs(stretch(source_signal, rate))
#         stretch_reset_action.append(warp_signal.numpy())
    
#     stretch_reset_actions[key] = stretch_reset_action

# ====================================================================
# create word
# ====================================================================
# WORDCOUNT = 250
# _charlabels = []
# wordindex = 0
# letterset = {}
# while True:
#     nWordLength = RandomFactor(2, 9, 'N', (9+2+1)/2, 2)   # 单词长度
    
#     label = []                                            # 生成单词
#     for i in range(nWordLength):
#         curletter = RandomFactor(0, 25, 'U')
#         label.append(curletter)

#     charlabel = int2char(label)
#     if charlabel not in _charlabels and charlabel not in dictionarys:
#         _charlabels.append(charlabel)
#         wordindex+=1
    
#     if wordindex == WORDCOUNT:
#         break


# =====================================================================
# synthe data
# =====================================================================
# datasets = []
# _intlabels = []
# for charlabel in _charlabels:
#     label = char2int(charlabel)
#     for j in range(10):                                   # 每个单词构建10个
#         w = np.zeros(shape=(114,0))
#         for letter in label:
#             w = np.concatenate((w, signals[j][letter]), axis=1)
            
#         _intlabels.append(label)
#         datasets.append(w)  

# for charlabel in _charlabels:
#     label = char2int(charlabel)
#     for j in range(10):
#         w = stretch_signals[j][label[0]]
#         for index in range(1, len(label)):
#             rn = letters_end[index-1] + letters_start[index]
#             rn = letters_end[label[index-1]] + letters_start[label[index]]
#             if rn != 'UU' and rn != 'DD':
#                 reset = reset_actions[rn][j]
#                 next_w = signals[j][label[index]]

#                 overlap_cols_pre = RandomOverlap(0.3, 0.5, reset.shape[1])
#                 overlap_cols_last = RandomOverlap(0.3, 0.5, reset.shape[1])

#                 w_1 = w[:, :w.shape[1] - overlap_cols_pre]
#                 w_2 = w[:, w.shape[1]-overlap_cols_pre:]
#                 reset_1 = reset[:,:overlap_cols_pre]
#                 reset_2 = reset[:,overlap_cols_pre:reset.shape[1] - overlap_cols_last]
#                 reset_3 = reset[:,reset.shape[1] - overlap_cols_last:]
#                 next_w_1 = next_w[:, :overlap_cols_last]
#                 next_w_2 = next_w[:, overlap_cols_last:]

#                 overlap_mat1 = merge(w_2, reset_1)
#                 overlap_mat2 = merge(reset_3, next_w_1)
#                 # overlap_mat1 = (w_2 + reset_1)/2
#                 # overlap_mat2 = (reset_3 + next_w_1)/2

#                 w = np.concatenate((w_1, overlap_mat1, reset_2, overlap_mat2, next_w_2), axis=1)
            
#             else:
#                 next_w = signals[j][label[index]]
#                 overlap_cols = RandomOverlap(0.3, 0.5, min(38, next_w.shape[1]))

#                 w_1 = w[:, :w.shape[1] - overlap_cols]
#                 w_2 = w[:, w.shape[1]-overlap_cols:]

#                 next_w_1 = next_w[:, :overlap_cols]
#                 next_w_2 = next_w[:, overlap_cols:]

#                 overlap_mat = merge(w_2, next_w_1)
#                 # overlap_mat = (w_2 + next_w_1)/2
                
#                 w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)

#         _intlabels.append(label)
#         datasets.append(np.flip(w, axis=0))


# for i in range(20):
#     plt.figure(figsize=(8,6))
#     plt.title(int2char(_intlabels[i]))
#     plt.imshow(datasets[i])
#     plt.savefig("./img/synthe/" + str(i) + "_" + int2char(_intlabels[i]) + ".png")
#     plt.close()
# exit(0)


# =====================================================================
# segment data
# =====================================================================
# trainDatas = []
# trainLabels = []
# for i in range(len(datasets)):  
#     word = datasets[i]
#     if (((word.shape[1] - TrainParam.WINDOW_SIZE) / TrainParam.STRIDE) + 1 < len(_intlabels[i]) + 1):
#         pad_size = ((len(_intlabels[i])) * TrainParam.STRIDE + TrainParam.WINDOW_SIZE - word.shape[1])
#         pad_size = math.ceil(pad_size / 2)
#         word = np.pad(word, ((0,0),(pad_size, pad_size)), mode="constant")

#     index = 0
#     t = word.shape[1]
#     # 记录每个词切分后的序列
#     data = []
#     while (index + TrainParam.WINDOW_SIZE) <= t:
#         d = word[:, index:index+TrainParam.WINDOW_SIZE]
#         if d.shape[1] < TrainParam.WINDOW_SIZE:
#             d = np.pad(d, ((0,0), (0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
#         d = d.reshape(1, d.shape[0], d.shape[1])
#         data.append(d)
#         index += TrainParam.STRIDE

#     if((index + (TrainParam.WINDOW_SIZE // 2)) < t):
#         d = word[:, index:-1]
#         d = np.pad(d, ((0,0),(0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
#         d = d.reshape(1, d.shape[0], d.shape[1])
#         data.append(d)

#     data = np.array(data)
#     label = np.array(_intlabels[i])

#     if label.shape[0] <= data.shape[0]:
#         trainDatas.append(data)
#         trainLabels.append(label)


# trainDatas = np.array(trainDatas, dtype=np.ndarray)
# trainLabels = np.array(trainLabels, dtype=np.ndarray)


# if os.path.exists(TrainParam.DATASET_PATH) is False:
#     os.mkdir(TrainParam.DATASET_PATH)

# np.save(os.path.join(TrainParam.DATASET_PATH, "train_datas.npy"), trainDatas)
# np.save(os.path.join(TrainParam.DATASET_PATH, "train_labels.npy"), trainLabels)

# print("train dataset synthe successful!")

# ===================================
# test dataset
# ===================================


# _intlabels = []
# datasets = []
 
# for i in range(len(dictionarys)):
#     label = char2int(dictionarys[i])
#     for j in range(10):
#         w = signals[j][label[0]]
#         for index in range(1, len(label)):
#             rn = letters_end[label[index-1]] + letters_start[label[index]]
#             if rn != 'UU' and rn != 'DD':
#                 reset = reset_actions[rn][j]
#                 next_w = signals[j][label[index]]

#                 overlap_cols_pre = RandomOverlap(0.3, 0.5, reset.shape[1])
#                 overlap_cols_last = RandomOverlap(0.3, 0.5, reset.shape[1])

#                 w_1 = w[:, :w.shape[1] - overlap_cols_pre]
#                 w_2 = w[:, w.shape[1]-overlap_cols_pre:]
#                 reset_1 = reset[:,:overlap_cols_pre]
#                 reset_2 = reset[:,overlap_cols_pre:reset.shape[1] - overlap_cols_last]
#                 reset_3 = reset[:,reset.shape[1] - overlap_cols_last:]
#                 next_w_1 = next_w[:, :overlap_cols_last]
#                 next_w_2 = next_w[:, overlap_cols_last:]

#                 overlap_mat1 = merge(w_2, reset_1)
#                 overlap_mat2 = merge(reset_3, next_w_1)
#                 # overlap_mat1 = (w_2 + reset_1)/2
#                 # overlap_mat2 = (reset_3 + next_w_1)/2

#                 w = np.concatenate((w_1, overlap_mat1, reset_2, overlap_mat2, next_w_2), axis=1)
            
#             else:
#                 next_w = signals[j][label[index]]
#                 overlap_cols = RandomOverlap(0.3, 0.5, min(38, next_w.shape[1]))

#                 w_1 = w[:, :w.shape[1] - overlap_cols]
#                 w_2 = w[:, w.shape[1]-overlap_cols:]

#                 next_w_1 = next_w[:, :overlap_cols]
#                 next_w_2 = next_w[:, overlap_cols:]

#                 overlap_mat = merge(w_2, next_w_1)
#                 # overlap_mat = (w_2 + next_w_1)/2
                
#                 w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)

#         _intlabels.append(label)
#         # w = np.pad(w, ((0,0),(4,4)), mode="constant")
#         datasets.append(np.flip(w,axis=0))


# path = "./img/ZCL_original_synthe"
# for i in range(len(datasets)):
#     # _, _, word = SignalProcess.preprocess(words[i], nperseg=TrainParam.NPERSEG, noverlap=TrainParam.NOVERLAP, FS=TrainParam.FS, log_flag=True, flip_flag=True)
#     word = datasets[i]
#     label = int2char(_intlabels[i])
#     if os.path.exists(os.path.join(path, label)) is False:
#         os.makedirs(os.path.join(path, label))

#     plt.figure()
#     plt.title(label + str(i))
#     plt.imshow(word)
#     plt.savefig(os.path.join(os.path.join(path, label), str(i) + "_" + label + ".png"))
#     plt.close()


# exit(0)
_intlabels = []
datasets = []
COUNT = 50
for i in range(len(dictionarys)):
    label = char2int(dictionarys[i])
    charlabel = dictionarys[i]
    for j in range(COUNT):
        k = RandomFactor(Low=0, High=9, flag='U')
        w = torch.from_numpy(np.ascontiguousarray(signals[k][label[0]]))
        if w.shape[1] < factor_down[charlabel[0]] or w.shape[1] > factor_up[charlabel[0]]:
            warp_len = RandomFactor(Low=factor_down[charlabel[0]], High=factor_up[charlabel[0]], flag='U')
            rate = w.shape[1] / warp_len
            w = torch.abs(stretch(w, rate))

        for index in range(1, len(label)):
            rn = letters_end[label[index-1]] + letters_start[label[index]]
            k = RandomFactor(Low=0, High=9, flag='U')
            if rn != 'UU' and rn != 'DD':
                rk = RandomFactor(Low=0, High=9, flag='U')
                reset = torch.from_numpy(np.ascontiguousarray(reset_actions[rn][rk]))
                if reset.shape[1] < reset_factor_down[rn] or reset.shape[1] > reset_factor_up[rn]:
                    reset_warp_len = RandomFactor(Low=reset_factor_down[rn], High=reset_factor_up[rn], flag='U')
                    reset_rate = reset.shape[1] / reset_warp_len
                    reset = torch.abs(stretch(reset, reset_rate))
                
                next_w = torch.from_numpy(np.ascontiguousarray(signals[k][label[index]]))
                if next_w.shape[1] < factor_down[charlabel[index]] or next_w.shape[1] > factor_up[charlabel[index]]:
                    warp_len = RandomFactor(Low=factor_down[charlabel[index]], High=factor_up[charlabel[index]], flag='U')
                    rate = next_w.shape[1] / warp_len
                    next_w = torch.abs(stretch(next_w, rate))

                overlap_cols_pre = RandomOverlap(0.45, 0.5, reset.shape[1])
                overlap_cols_last = RandomOverlap(0.45, 0.5, reset.shape[1])

                w_1 = w[:, :w.shape[1] - overlap_cols_pre]
                w_2 = w[:, w.shape[1]-overlap_cols_pre:]

                reset_1 = reset[:,:overlap_cols_pre]
                reset_2 = reset[:,overlap_cols_pre:reset.shape[1] - overlap_cols_last]
                reset_3 = reset[:,reset.shape[1] - overlap_cols_last:]

                next_w_1 = next_w[:, :overlap_cols_last]
                next_w_2 = next_w[:, overlap_cols_last:]

                overlap_mat1 = merge(w_2, reset_1)
                overlap_mat2 = merge(reset_3, next_w_1)

                w = torch.cat((w_1, overlap_mat1, reset_2, overlap_mat2, next_w_2), axis=1)
            else:
                next_w = torch.from_numpy(np.ascontiguousarray(signals[k][label[index]]))
                if next_w.shape[1] < factor_down[charlabel[index]] or next_w.shape[1] > factor_up[charlabel[index]]:
                    warp_len = RandomFactor(Low=factor_down[charlabel[index]], High=factor_up[charlabel[index]], flag='U')
                    rate = next_w.shape[1] / warp_len
                    next_w = torch.abs(stretch(next_w, rate))

                overlap_cols = RandomFactor(-4, 8, flag='U')

                if overlap_cols < 0:
                    overlap_w = torch.zeros(size=(114, abs(overlap_cols))).type_as(next_w)
                    w = torch.cat((w, overlap_w, next_w), axis=1)
                else:
                    w_1 = w[:, :w.shape[1] - overlap_cols]
                    w_2 = w[:, w.shape[1]-overlap_cols:]

                    next_w_1 = next_w[:, :overlap_cols]
                    next_w_2 = next_w[:, overlap_cols:]

                    overlap_mat = merge(w_2, next_w_1)

                    w = torch.cat((w_1, overlap_mat, next_w_2), axis=1)
        
        _intlabels.append(label)
        w = w.numpy()
        w = np.pad(w, ((0,0),(4,4)), mode="constant")
        datasets.append(np.flip(w,axis=0))



# path = "./img/ZCL_original_synthe_v4"
# if os.path.exists(path) is False:
#     os.makedirs(path)

# for i in range(len(datasets)):
#     word = datasets[i]
#     label = int2char(_intlabels[i])

#     if os.path.exists(os.path.join(path, label)) is False:
#         os.makedirs(os.path.join(path, label))
    
#     plt.figure()
#     plt.title(label + str(i))
#     plt.imshow(word)
#     plt.savefig(os.path.join(os.path.join(path, label), str(i) + "_" + label + ".png"))
#     plt.close()

# exit(0)


testDatas = []
testLabels = []
for i in range(len(datasets)):  
    word = datasets[i]
    flag = False
    if (((word.shape[1] - TrainParam.WINDOW_SIZE) / TrainParam.STRIDE) + 1 < len(_intlabels[i])):
        pad_size = ((len(_intlabels[i]) - 1) * TrainParam.STRIDE + TrainParam.WINDOW_SIZE - word.shape[1])
        pad_size = math.ceil(pad_size / 2)
        word = np.pad(word, ((0,0),(pad_size, pad_size)), mode="constant")
        flag = True

    index = 0
    t = word.shape[1]
    # 记录每个词切分后的序列
    data = []
    while (index + TrainParam.WINDOW_SIZE) <= t:
        d = word[:, index:index+TrainParam.WINDOW_SIZE]
        # if d.shape[1] < TrainParam.WINDOW_SIZE:
        #     d = np.pad(d, ((0,0), (0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
        d = d.reshape(1, d.shape[0], d.shape[1])
        data.append(d)
        index += TrainParam.STRIDE

    # if((index + (TrainParam.WINDOW_SIZE // 2)) < t):
    if(flag is False):
        d = word[:, index:]
        if(d.shape[1] < TrainParam.WINDOW_SIZE):
            d = np.pad(d, ((0,0),(0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
            d = d.reshape(1, d.shape[0], d.shape[1])
            data.append(d)

    data = np.array(data)
    label = np.array(_intlabels[i])

    if label.shape[0] <= data.shape[0]:
        testDatas.append(data)
        testLabels.append(label)


testDatas = np.array(testDatas, dtype=np.ndarray)
testLabels = np.array(testLabels, dtype=np.ndarray)


# np.save(os.path.join(TrainParam.DATASET_PATH, "test_datas.npy"), testDatas)
# np.save(os.path.join(TrainParam.DATASET_PATH, "test_labels.npy"), testLabels)
if os.path.exists("../DATA/synthe_dataset_randomcreate_reset_v10") is False:
    os.mkdir("../DATA/synthe_dataset_randomcreate_reset_v10")
np.save(os.path.join("../DATA/synthe_dataset_randomcreate_reset_v10", "test_datas.npy"), testDatas)
np.save(os.path.join("../DATA/synthe_dataset_randomcreate_reset_v10", "test_labels.npy"), testLabels)

print("test dataset synthe successful!")

