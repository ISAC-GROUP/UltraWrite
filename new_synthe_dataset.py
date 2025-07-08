# =============================================
#   @File Name : new_synthe_dataset.py
#   @Author : Acol
#   @Version : 1.1
#   @Created Time : 2022/8/18 四
# =============================================

import numpy as np
import os
import random

import TrainParam
import math
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch

root_path = "../DATA/letter/ZCL/after_process_v12"
# reset_root_path = "../DATA/letter/ZCL/after_process_v4"
save_path = "../DATA/synthe_dataset_reset_diff_speed_v16_3"
if os.path.exists(save_path) is False:
    os.mkdir(save_path)
# ======================================================
#              Read Source Letter Signal
# ======================================================
#   @ Args:
#       Data Path : "../DATA/letter/ZCL/DiffSpeed"
#       Source Letter Dataset Shape : [30, 26]      
#       After Warp Letter Dataset shape : [150, 26] 
#       Source Reset Dataset Shape : [30, 4]
#       After Warp Reset Dataset shae : [150, 4]
#   
#   @Comment: each letter or reset have 3 speed 
#             format and 2 size format. at the 
#             same time, one group of speed and 
#             size will write 5 time. finally,
#             source signal will have [3 * 2 * 5]
#             circle signal(A ~ Z).
# 

letter_signal_paths = []
letter_index_paths = []
reset_signal_paths = []
reset_index_paths = []

for root, dir, files in os.walk(root_path):
    for file in files:
        # if "LF" in file or "SF" in file:
        #     continue
        if "letter" in file:
            if "signal" in file:
                letter_signal_paths.append(os.path.join(root, file))
                # continue
            elif "index" in file:
                letter_index_paths.append(os.path.join(root, file))
                # continue
        else:
            if "signal" in file:
                reset_signal_paths.append(os.path.join(root, file))
            elif "index" in file:
                reset_index_paths.append(os.path.join(root, file))

letter_signal_paths.sort()
letter_index_paths.sort()
reset_signal_paths.sort()
reset_index_paths.sort()

# letter_signal_paths = np.array(letter_signal_paths)
# print(letter_signal_paths)
# letter_index_paths = np.array(letter_index_paths)
# print(letter_index_paths)

# reset_signal_paths = np.array(reset_signal_paths)
# print(reset_signal_paths)
# reset_index_paths = np.array(reset_index_paths)
# print(reset_index_paths)
# exit(0)


diff_speed_letter_signals = []
diff_speed_reset_signals = []
for i in range(len(letter_signal_paths)):

    # letter signal 
    letter_signal_path = letter_signal_paths[i]
    letter_index_path = letter_index_paths[i]

    letter_signal = np.load(letter_signal_path)
    letter_index = np.load(letter_index_path)

    letter = []                                          # 同一轮的数据保存为一个list，每个元素为一个ndarray，从A到Z;
    for idx in letter_index:
        letter.append(letter_signal[:, idx[0]:idx[1]])

    diff_speed_letter_signals.append(letter)

    # reset signal
    # each circle of reset action name is [UD, MU, MD, DU]
    reset_signal_path = reset_signal_paths[i]
    reset_index_path = reset_index_paths[i]

    reset_signal = np.load(reset_signal_path)
    reset_index = np.load(reset_index_path)

    reset = []

    for idx in reset_index:
        reset.append(reset_signal[:, idx[0]:idx[1]])
    
    diff_speed_reset_signals.append(reset)

# exit(0)
# =========================================================
#                     Warping Signal
# =========================================================
#
#   @Comment: each circle letter or reset signal will be
#             warp into 5 new circle. each circle's warp
#             rate is [1.2, 1.1, 1.0, 0.9, 0.8].
# 

stretch = T.TimeStretch(n_freq=114)

diff_speed_letter_signals_after_warp = []
# rates = [1.0]
# rates = [1.05, 1.0, 0.95]
# rates = [1.1, 1.05, 1.0, 0.95, 0.9]
# rates = [1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85]
rates = [1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8]
for letter_signals in diff_speed_letter_signals:        # letter_signals is list, and element is ndarray of A to Z.
    for rate in rates:
        if rate == 1.0:
            diff_speed_letter_signals_after_warp.append(letter_signals)
        else:
            after_warp_letter = []
            for letter in letter_signals:
                letter = torch.from_numpy(np.ascontiguousarray(letter))
                letter = torch.abs(stretch(letter, rate))
                after_warp_letter.append(letter.numpy())

            diff_speed_letter_signals_after_warp.append(after_warp_letter)

diff_speed_reset_signals_after_warp = []
for reset_signals in diff_speed_reset_signals:
    for rate in rates:
        if rate == 1.0:
            diff_speed_reset_signals_after_warp.append(reset_signals)
        else:
            after_warp_reset = []
            for reset in reset_signals:
                reset = torch.from_numpy(np.ascontiguousarray(reset))
                reset = torch.abs(stretch(reset, rate))
                after_warp_reset.append(reset.numpy())

            diff_speed_reset_signals_after_warp.append(after_warp_reset)

DIFF_SPEED_TIME = len(diff_speed_letter_signals_after_warp)
# print(DIFF_SPEED_TIME)
# print(len(letter_signal_paths))
# print(len(letter_index_paths))
# print(len(reset_signal_paths))
# print(len(reset_index_paths))
# exit(0)
# print(len(diff_speed_letter_signals_after_warp))
# print(len(diff_speed_reset_signals_after_warp))
# exit(0)
# ===========================================
#            Detial Of Letter 
# ===========================================
letters_start = [
    'U', 'U', 'U', 'U', 'U', 'U', 'U',\
    'U', 'U', 'U', 'U', 'U', 'U', 'U',\
    'D', 'U', 'U', 'U', 'U', 'U',\
    'U', 'U', 'U', 'U', 'U', 'U'
]
letters_end = [
    'M', 'D', 'M', 'D', 'M', 'D', 'M',\
    'D', 'D', 'M', 'D', 'D', 'U', 'D', \
    'D', 'M', 'D', 'D', 'D', 'D',\
    'U', 'U', 'D', 'D', 'D', 'D'
]

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

resetName2Index = {"UD":0, "MU":1, "MD":2, "DU":3}

# ===========================================
#               Dictionary
# ===========================================
# dictionarys = [
#     'AM', 'AND', 'ARE', 'BACK',\
#     'BE', 'BUT', 'CAN', 'COME',\
#     'DO', 'EIGHT', 'FAMILY', 'FATHER',\
#     'FIVE', 'FOUR', 'GIVE', 'GOOD',\
#     'HAVE', 'HE', 'HELLO', 'HOW',\
#     'IS', 'JOB', 'JUST', 'MAKE',\
#     'MAN', 'ME', 'MOTHER', 'NEXT',\
#     'NINE', 'ONE', 'PROJECT', 'PUT',\
#     'QUIT', 'REQUIRE', 'SAY', 'SEVEN',\
#     'SIX', 'SIZE', 'THAT', 'THE', \
#     'THEY', 'THINK', 'THREE', 'TWO',\
#     'WHAT', 'WHERE', 'WHY', 'WORLD',\
#     'YOU', 'ZERO'
# ]
dictionarys = [
    'AM','ARE','BUT','CAN','DO',\
    'FAMILY','GIVE','HELLO','HOW','IS',\
    'JOB','MAKE','NEXT', 'PERSON', 'PUT',\
    'QUIT','SAY','THAT','THE','WHAT',\
    'WHERE','WHY','WORLD','YOU','ZERO']

# dictionarys = [
#     'AM','ARE','HE','HELLO','IS',\
#     'THE','WORLD','YOU']


# =======================
#    Using Function
# =======================

def RandomFactor(Low:int = 0, High:int = 25, flag:str='N', mean:int = 1, sigma:int = 2) -> int:
    '''
        @Args
        -----
            Low(int): the min of distribute
            High(int): the max of distribute
            flag(str): select which distribute will be use. ['N', 'U']
            mean(int): A options. when flag is N. this arg will be use
            sigma(int): A options. same as mean
    
        @Return
        -------
            nResult(int): random number.
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

def merge(mat1:np.ndarray, mat2:np.ndarray)-> np.ndarray:
    if mat1.shape[1] == 0:
        return mat1
    mat = np.zeros_like(mat1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[i][j] = max(mat1[i][j], mat2[i][j])
    return mat

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


# ========================================================
#             Create Random Charactor Assembly
# ========================================================
#   @Args:
#       WORDCOUMT: how many random word will be 
#               create.
# 
#   @Comment: This task is create some random character
#             assembly, such as AB, CACD, DGHGHR. The 
#             len of them from [2 ~ 9]. In this task, 
#             will create 250 random character assembly
#             and all of them are different than word 
#             in dictionarys.
#
#

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

gramspath = "./resource/Grams_v6.txt"

_charlabels = []
readerFd = open(gramspath, mode='r')

while True:
    charlabel = readerFd.readline()
    if charlabel == '':
        break
    charlabel = charlabel.strip("\n")
    _charlabels.append(charlabel)

readerFd.close()


# ============================================================
#                   Synthe Random Word Dataset
# ============================================================
# 
#   @Comment: This task will synthe train data according to
#             the charlabels which create in "Create Random 
#             Character Assembly". Each character Assembly 
#             will be synthe in DIFF_SPEED_TIME times. The 
#             overlap between current letter, last letter
#             and the reset action of them will random crea-
#             te by using function(RandomOverlap). And using 
#             function(merge) to merge them.
#
#

# use this
datasets = []
_intlabels = []

for charlabel in _charlabels:
    intlabel = char2int(charlabel)
    for j in range(DIFF_SPEED_TIME):
        w = diff_speed_letter_signals_after_warp[j][intlabel[0]]
        for index in range(1, len(intlabel)):
            rn = letters_end[intlabel[index-1]] + letters_start[intlabel[index]]
            if rn != "UU" and rn != "DD":
                reset = diff_speed_reset_signals_after_warp[j][resetName2Index[rn]]
                next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]

                overlap_cols_pre = RandomFactor(-30, 6, flag='U')
                overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])
                
                # overlap_cols_last = RandomFactor(-4, 4, flag='U') # 表示可以存在一定的间隙

                if overlap_cols_pre <= 0:
                    w_1 = w
                    w_2 = np.zeros(shape=(114, abs(overlap_cols_pre)), dtype=w.dtype)
                    reset_1 = np.zeros_like(w_2)
                    reset_2 = reset[:, :-overlap_cols_last]
                    reset_3 = reset[:, -overlap_cols_last:]
                
                else:
                    w_1 = w[:, :-overlap_cols_pre]
                    w_2 = w[:, -overlap_cols_pre:]
                    reset_1 = reset[:, :overlap_cols_pre]
                    reset_2 = reset[:, overlap_cols_pre:-overlap_cols_last]
                    reset_3 = reset[:, -overlap_cols_last:]                
            
                next_w_1 = next_w[:, :overlap_cols_last]
                next_w_2 = next_w[:, overlap_cols_last:]

                overlap_mat_1 = merge(w_2, reset_1)
                overlap_mat_2 = merge(reset_3, next_w_1)

                w = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)

            else:
                next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]
                overlap_cols = RandomFactor(-30, 4, flag='U') # 表示可以存在一定的间隙
                
                if overlap_cols <= 0:
                    overlap_w = np.zeros(shape=(114, abs(overlap_cols)), dtype=next_w.dtype)
                    w = np.concatenate((w, overlap_w, next_w), axis=1)
                    # w = np.concatenate((w, next_w), axis=1)
                else:
                    w_1 = w[:, :-overlap_cols]
                    w_2 = w[:, -overlap_cols:]

                    next_w_1 = next_w[:, :overlap_cols]
                    next_w_2 = next_w[:, overlap_cols:]

                    overlap_mat = merge(w_2, next_w_1)

                    w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)

        _intlabels.append(intlabel)
        w = np.pad(w, ((0,0),(4,4)), mode="constant")
        datasets.append(np.flip(w, axis=0))

# for charlabel in _charlabels:
#     intlabel = char2int(charlabel)
#     for j in range(DIFF_SPEED_TIME):
#         w = diff_speed_letter_signals_after_warp[j][intlabel[0]]
#         for index in range(1, len(intlabel)):
#             rn = letters_end[intlabel[index-1]] + letters_start[intlabel[index]]
#             if rn != "UU" and rn != "DD":
#                 reset = diff_speed_reset_signals_after_warp[j][resetName2Index[rn]]
#                 next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]

#                 overlap_cols_pre = RandomOverlap(0.2, 0.5, reset.shape[1])
#                 overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])

#                 w_1 = w[:, :-overlap_cols_pre]
#                 w_2 = w[:, -overlap_cols_pre:]

#                 reset_1 = reset[:,:overlap_cols_pre]
#                 reset_2 = reset[:,overlap_cols_pre:-overlap_cols_last]
#                 reset_3 = reset[:,-overlap_cols_last:]

#                 next_w_1 = next_w[:, :overlap_cols_last]
#                 next_w_2 = next_w[:, overlap_cols_last:]

#                 overlap_mat_1 = merge(w_2, reset_1)
#                 overlap_mat_2 = merge(reset_3, next_w_1)

#                 w = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)

#             else:
#                 next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]
#                 overlap_cols = RandomFactor(0, 4, flag='U') # 表示可以存在一定的间隙
                
#                 if overlap_cols == 0:
#                     # overlap_w = np.zeros(shape=(114, abs(overlap_cols)), dtype=next_w.dtype)
#                     # w = np.concatenate((w, overlap_w, next_w), axis=1)
#                     w = np.concatenate((w, next_w), axis=1)
#                 else:
#                     w_1 = w[:, :-overlap_cols]
#                     w_2 = w[:, -overlap_cols:]

#                     next_w_1 = next_w[:, :overlap_cols]
#                     next_w_2 = next_w[:, overlap_cols:]

#                     overlap_mat = merge(w_2, next_w_1)

#                     w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)

#         _intlabels.append(intlabel)
#         w = np.pad(w, ((0,0),(4,4)), mode="constant")
#         datasets.append(np.flip(w, axis=0))


# for i in range(0, len(datasets), 50):
#     data = datasets[i]
#     label = int2char(_intlabels[i])

#     plt.figure(figsize=(10, 6))
#     plt.imshow(data)
#     plt.title(label)
    
#     plt.savefig("./img/Diff_Speed/" + label + "_" + str(i+1) + ".png")
#     plt.close()

# exit(0)


# ========================================
#       Segment Random Word Dataset
# ========================================
trainDatas = []
trainLabels = []
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

    if(flag is False):
        d = word[:, index:]
        if(d.shape[1] < TrainParam.WINDOW_SIZE):
            d = np.pad(d, ((0,0),(0, TrainParam.WINDOW_SIZE - d.shape[1])), mode="constant")
            d = d.reshape(1, d.shape[0], d.shape[1])
            data.append(d)

    data = np.array(data)
    label = np.array(_intlabels[i])

    if label.shape[0] <= data.shape[0]:
        trainDatas.append(data)
        trainLabels.append(label)

print(len(trainDatas))
trainDatas = np.array(trainDatas, dtype=np.ndarray)
trainLabels = np.array(trainLabels, dtype=np.ndarray)

np.save(os.path.join(save_path, "synthe_train_datas.npy"), trainDatas)
np.save(os.path.join(save_path, "synthe_train_labels.npy"), trainLabels)
print("train dataset synthe successful!")
# del trainDatas
# del trainLabels

exit(0)
# ============================================
#       Synthe Dictionary Word Dataset
# ============================================
_intlabels = []
datasets = []

##############
#   v4
##############

for i in range(len(dictionarys)):
    intlabel = char2int(dictionarys[i])
    charlabel = dictionarys[i]
    for j in range(DIFF_SPEED_TIME):
        w = diff_speed_letter_signals_after_warp[j][intlabel[0]]
        for index in range(1, len(intlabel)):
            rn = letters_end[intlabel[index-1]] + letters_start[intlabel[index]]

            # 存在归位动作情况
            if(rn != 'UU' and rn != 'DD'):
                reset = diff_speed_reset_signals_after_warp[j][resetName2Index[rn]]
                next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]

                overlap_cols_pre = RandomFactor(-30, 6, flag='U')
                # overlap_cols_pre = RandomOverlap(0.2, 0.5, reset.shape[1])
                overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])
                
                # overlap_cols_last = RandomFactor(-4, 4, flag='U') # 表示可以存在一定的间隙

                if overlap_cols_pre <= 0:
                    w_1 = w
                    w_2 = np.zeros(shape=(114, abs(overlap_cols_pre)), dtype=w.dtype)
                    reset_1 = np.zeros_like(w_2)
                    reset_2 = reset[:, :-overlap_cols_last]
                    reset_3 = reset[:, -overlap_cols_last:]
                
                else:
                    w_1 = w[:, :-overlap_cols_pre]
                    w_2 = w[:, -overlap_cols_pre:]
                    reset_1 = reset[:, :overlap_cols_pre]
                    reset_2 = reset[:, overlap_cols_pre:-overlap_cols_last]
                    reset_3 = reset[:, -overlap_cols_last:]                
            
                next_w_1 = next_w[:, :overlap_cols_last]
                next_w_2 = next_w[:, overlap_cols_last:]

                # overlap_cols_pre = RandomOverlap(0.2, 0.5, reset.shape[1])
                # # overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])
                # overlap_cols_last = RandomFactor(-4, 4, flag='U') # 表示可以存在一定的间隙

                # w_1 = w[:, :-overlap_cols_pre]
                # w_2 = w[:, -overlap_cols_pre:]

                # reset_1 = reset[:,:overlap_cols_pre]
                # if overlap_cols_last <= 0:
                #     reset_2 = reset[:, overlap_cols_pre:]
                #     reset_3 = np.zeros(shape=(114, abs(overlap_cols_last)), dtype=w_2.dtype)
                #     next_w_1 = np.zeros_like(reset_3)
                #     next_w_2 = next_w

                # else:
                #     reset_2 = reset[:,overlap_cols_pre:-overlap_cols_last]
                #     reset_3 = reset[:,-overlap_cols_last:]

                #     next_w_1 = next_w[:, :overlap_cols_last]
                #     next_w_2 = next_w[:, overlap_cols_last:]

                overlap_mat_1 = merge(w_2, reset_1)
                overlap_mat_2 = merge(reset_3, next_w_1)

                w = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)

            # 不存在归位动作情况
            else:
                next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]
                overlap_cols = RandomFactor(-30, 4, flag='U') # 表示可以存在一定的间隙
                # overlap_cols = RandomOverlap(0.2, 0.5, reset.shape[1])

                if overlap_cols <= 0:
                    overlap_w = np.zeros(shape=(114, abs(overlap_cols)), dtype=next_w.dtype)
                    w = np.concatenate((w, overlap_w, next_w), axis=1)
                    # w = np.concatenate((w, next_w), axis=1)
                else:
                    w_1 = w[:, :-overlap_cols]
                    w_2 = w[:, -overlap_cols:]

                    next_w_1 = next_w[:, :overlap_cols]
                    next_w_2 = next_w[:, overlap_cols:]

                    overlap_mat = merge(w_2, next_w_1)

                    w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)
        
        _intlabels.append(intlabel)
        w = np.pad(w, ((0,0),(4,4)), mode="constant")
        datasets.append(np.flip(w, axis=0))


###############
#     v5
###############
# TIMES = 20
# for i in range(len(dictionarys)):
#     intlabel = char2int(dictionarys[i])
#     charlabel = dictionarys[i]
#     for j in range(DIFF_SPEED_TIME):
#         for k in range(TIMES):
#             w = diff_speed_letter_signals_after_warp[j][intlabel[0]]
#             for index in range(1, len(intlabel)):
#                 rn = letters_end[intlabel[index-1]] + letters_start[intlabel[index]]

#                 # 存在归位动作情况
#                 if(rn != 'UU' and rn != 'DD'):
#                     reset = diff_speed_reset_signals_after_warp[j][resetName2Index[rn]]
#                     next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]

#                     overlap_cols_pre = RandomFactor(-30, 6, flag='U')
#                     overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])
                    
#                     # overlap_cols_last = RandomFactor(-4, 4, flag='U') # 表示可以存在一定的间隙

#                     if overlap_cols_pre <= 0:
#                         w_1 = w
#                         w_2 = np.zeros(shape=(114, abs(overlap_cols_pre)), dtype=w.dtype)
#                         reset_1 = np.zeros_like(w_2)
#                         reset_2 = reset[:, :-overlap_cols_last]
#                         reset_3 = reset[:, -overlap_cols_last:]
                    
#                     else:
#                         w_1 = w[:, :-overlap_cols_pre]
#                         w_2 = w[:, -overlap_cols_pre:]
#                         reset_1 = reset[:, :overlap_cols_pre]
#                         reset_2 = reset[:, overlap_cols_pre:-overlap_cols_last]
#                         reset_3 = reset[:, -overlap_cols_last:]                
                
#                     next_w_1 = next_w[:, :overlap_cols_last]
#                     next_w_2 = next_w[:, overlap_cols_last:]


#                     overlap_mat_1 = merge(w_2, reset_1)
#                     overlap_mat_2 = merge(reset_3, next_w_1)

#                     w = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)

#                 # 不存在归位动作情况
#                 else:
#                     next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]
#                     overlap_cols = RandomFactor(-30, 4, flag='U') # 表示可以存在一定的间隙
                    
#                     if overlap_cols <= 0:
#                         overlap_w = np.zeros(shape=(114, abs(overlap_cols)), dtype=next_w.dtype)
#                         w = np.concatenate((w, overlap_w, next_w), axis=1)
#                         # w = np.concatenate((w, next_w), axis=1)
#                     else:
#                         w_1 = w[:, :-overlap_cols]
#                         w_2 = w[:, -overlap_cols:]

#                         next_w_1 = next_w[:, :overlap_cols]
#                         next_w_2 = next_w[:, overlap_cols:]

#                         overlap_mat = merge(w_2, next_w_1)

#                         w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)
        
#             _intlabels.append(intlabel)
#             w = np.pad(w, ((0,0),(4,4)), mode="constant")
#             datasets.append(np.flip(w, axis=0))

# print(len(_intlabels))
# exit(0)

# for i in range(len(dictionarys)):
#     intlabel = char2int(dictionarys[i])
#     charlabel = dictionarys[i]
#     for j in range(DIFF_SPEED_TIME):
#         w = diff_speed_letter_signals_after_warp[j][intlabel[0]]
#         for index in range(1, len(intlabel)):
#             rn = letters_end[intlabel[index-1]] + letters_start[intlabel[index]]

#             # 存在归位动作情况
#             if(rn != 'UU' and rn != 'DD'):
#                 reset = diff_speed_reset_signals_after_warp[j][resetName2Index[rn]]
#                 next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]

#                 overlap_cols_pre = RandomOverlap(0.2, 0.5, reset.shape[1])
#                 overlap_cols_last = RandomOverlap(0.2, 0.5, reset.shape[1])

#                 w_1 = w[:, :-overlap_cols_pre]
#                 w_2 = w[:, -overlap_cols_pre:]

#                 reset_1 = reset[:,:overlap_cols_pre]
#                 reset_2 = reset[:,overlap_cols_pre:-overlap_cols_last]
#                 reset_3 = reset[:,-overlap_cols_last:]

#                 next_w_1 = next_w[:, :overlap_cols_last]
#                 next_w_2 = next_w[:, overlap_cols_last:]

#                 overlap_mat_1 = merge(w_2, reset_1)
#                 overlap_mat_2 = merge(reset_3, next_w_1)

#                 w = np.concatenate((w_1, overlap_mat_1, reset_2, overlap_mat_2, next_w_2), axis=1)

#             # 不存在归位动作情况
#             else:
#                 next_w = diff_speed_letter_signals_after_warp[j][intlabel[index]]
#                 overlap_cols = RandomFactor(0, 4, flag='U') # 表示可以存在一定的间隙
                
#                 if overlap_cols == 0:
#                     # overlap_w = np.zeros(shape=(114, abs(overlap_cols)), dtype=next_w.dtype)
#                     # w = np.concatenate((w, overlap_w, next_w), axis=1)
#                     w = np.concatenate((w, next_w), axis=1)
#                 else:
#                     w_1 = w[:, :-overlap_cols]
#                     w_2 = w[:, -overlap_cols:]

#                     next_w_1 = next_w[:, :overlap_cols]
#                     next_w_2 = next_w[:, overlap_cols:]

#                     overlap_mat = merge(w_2, next_w_1)

#                     w = np.concatenate((w_1, overlap_mat, next_w_2), axis=1)
        
#         _intlabels.append(intlabel)
#         w = np.pad(w, ((0,0),(4,4)), mode="constant")
#         datasets.append(np.flip(w, axis=0))

# for i in range(0, len(datasets), 50):
#     data = datasets[i]
#     label = int2char(_intlabels[i])

#     plt.figure(figsize=(10, 6))
#     plt.imshow(data)
#     plt.title(label)
    
#     plt.savefig("./img/Diff_Speed/" + label + "_" + str(i+1) + ".png")
#     plt.close()

# exit(0)


# =============================================
#       Segment Dictionary Word Dataset
# =============================================
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

print(len(testDatas))

testDatas = np.array(testDatas, dtype=np.ndarray)
testLabels = np.array(testLabels, dtype=np.ndarray)


np.save(os.path.join(save_path, "synthe_test_datas.npy"), testDatas)
np.save(os.path.join(save_path, "synthe_test_labels.npy"), testLabels)
print("test dataset synthe successful!")