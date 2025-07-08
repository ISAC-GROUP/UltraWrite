
import numpy as np
import torch
import SignalProcess
import matplotlib.pyplot as plt
import os
import DataIO, SignalProcess

# ROOT_PATH = '../../DATA/letter_diff_speed'
# SAVE_PATH = '../../DATA/letter/ZCL/after_process_v13'
ROOT_PATH = "../../DATA/word/ZPZ/WORD_D_bu/"
SAVE_PATH = "../../DATA/ALL/final_real_word/ZPZ_WORD_bu"

if os.path.exists(SAVE_PATH) is False:
    os.mkdir(SAVE_PATH)

# print("yes")
# exit(0)
# patterns = ['LF', 'LN', 'LS', 'SF', 'SN', 'SS']
patterns = ["NLF", "NLN", "NLS"]
_int2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
    7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
    14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
    20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
}

# @Comment:
#  
# LF -> large size and fast speed
# LN -> large size and normal speed
# LS -> large size and slow speed
# SF -> small size and fast speed
# SN -> small size and normal speed
# SS -> small size and slow speed

# Each pattern will keep in a document. and pattern name
# is the document name. and each document will have 5 
# source wav signal. 

# letter written order:
# A O B C D E F G
# H I J K L M N
# O P Q R S T
# U V W X Y Z
# The reason why set written order like this is that we can
# get the reset in the same circle. (for same size and speed)

# letter_A2Z_indexs = [
#     0, 1, 2, 3, 4, 5, 6,\
#     7, 8, 9, 10, 11, 13, 12,\
#     15, 14, 16, 17, 18, 19, 20,\
#     21, 22, 23, 24, 25 
# ]

letter_A2Z_indexs_s = [
    0, 1, 2, 3, 4, 5, 6,\
    7, 8, 9, 10, 11, 13, 12,\
    15, 26, 16, 17, 18, 19, 20,\
    21, 22, 23, 24, 25 
]

reset_UD_MU_MD_DU_index = [14, 1, 15, 16]

def modify(signal, f):
    shape = signal.shape
    main_fq_each_window = np.zeros(shape=shape[1])
    for j in range(shape[1]):
        current_window = signal[:,j]
        main_fq_each_window[j] = SignalProcess.cal_main_fq(current_window, f)


    bindex = 0
    eindex = shape[1]-1
    # while(bindex+1 < shape[1] and main_fq_each_window[bindex+1] == 19000):
    #     bindex+=1
    
    while(eindex-1 >= 0 and main_fq_each_window[eindex-1] == 19000):
        eindex-=1
    signal = signal[:, bindex:eindex]

    return signal

def main0():
    # for pattern in patterns:
    pattern = "LF4"
    letters, resets, f = SignalProcess.split_letter(os.path.join(ROOT_PATH, pattern))

    # signals is a list with 5 circle data.
    # subcircle is a list to, with 27 letter, O have twice.
    for rank in range(len(letters)):
        # save letter signal
        subcircle = letters[rank]
        # save_letter = subcircle[letter_A2Z_indexs[0]]
        save_letter = subcircle[letter_A2Z_indexs_s[0]]
        save_letter = modify(save_letter, f)
        count = save_letter.shape[1]
        indexs = [np.array([0, count])]
        for i in range(1, len(letter_A2Z_indexs_s)):
            # letter = subcircle[letter_A2Z_indexs[i]]
            letter = subcircle[letter_A2Z_indexs_s[i]]
            letter = modify(letter, f)
            indexs.append(np.array([count, count + letter.shape[1]]))
            count+=letter.shape[1]
            save_letter = np.concatenate((save_letter, letter), axis=1)
        
        indexs = np.array(indexs)

        np.save(os.path.join(SAVE_PATH, "letter_rank_" + str(rank+1) + "signal.npy"), save_letter)
        np.save(os.path.join(SAVE_PATH, "letter_rank_" + str(rank+1) + "index.npy"), indexs)
        
        # save reset signal
        subreset = resets[rank]
        save_reset = subreset[reset_UD_MU_MD_DU_index[0]]
        save_reset = modify(save_reset, f)
        reset_count = save_reset.shape[1]
        reset_indexs = [np.array([0, reset_count])]

        for i in range(1, len(reset_UD_MU_MD_DU_index)):
            reset = subreset[reset_UD_MU_MD_DU_index[i]]
            reset = modify(reset, f)
            reset_indexs.append(np.array([reset_count, reset_count+reset.shape[1]]))
            reset_count += reset.shape[1]
            save_reset = np.concatenate((save_reset, reset), axis=1)
        
        reset_indexs = np.array(reset_indexs)
        np.save(os.path.join(SAVE_PATH, "reset_rank_" + str(rank+1) + "signal.npy"), save_reset)
        np.save(os.path.join(SAVE_PATH, "reset_rank_" + str(rank+1) + "index.npy"), reset_indexs)

# SignalProcess.main(os.path.join(ROOT_PATH, "LF4/ele2fifth"))
# main0()

    
# word_patterns = ["AM", "ARE", "BUT", "CAN", "DO", "FAMILY", "HELLO", "HOW",\
#      "IS", "JOB", "MAKE", "NEXT", "PERSON", "PUT", "SAY", "THAT","THE", "WHAT",\
#     "WHERE", "WHY", "YOU", "QUIT", "GIVE", "ZERO", "WORLD"]


word_patterns = ["AM", "BUT", "CAN", "HELLO", "HOW", "IS", "JOB", "MAKE", "NEXT", "PERSON", "QUIT", "SAY",\
                 "THAT","THE","WHAT", "WHERE","WORLD","YOU"]

# 
def main2(word_pattern):
    words, f = SignalProcess.split_word(os.path.join(ROOT_PATH, word_pattern))

    subcircle = []
    # subts = []

    for word in words:
        # subcircle += word[:-5]
        subcircle += word

    for i in range(len(subcircle)):
        save_word = subcircle[i]
        save_word = modify(save_word, f)

        np.save(os.path.join(SAVE_PATH, word_pattern + "_" + str(i+1) + ".npy"), save_word)

for word_pattern in word_patterns:
    main2(word_pattern)


def main3():
    root_path = "../../DATA/letter/ZCL/after_process_v13/"
    save_path = "../../DATA/letter/ZCL/after_process_v14/"

    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    letter_signal_paths = []
    letter_index_paths = []
    reset_signal_paths = []
    reset_index_paths = []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            pt = root + "/" + file
            if "letter" in file:
                if "signal" in file:
                    letter_signal_paths.append(pt)
                elif "index" in file:
                    letter_index_paths.append(pt)
            else:
                if "signal" in file:
                    reset_signal_paths.append(pt)
                elif "index" in file:
                    reset_index_paths.append(pt)

    letter_signal_paths.sort()
    letter_index_paths.sort()
    reset_signal_paths.sort()
    reset_index_paths.sort()


    letters = [[],[],[],[],[],[],[],\
                [],[],[],[],[],[],[],\
                [],[],[],[],[],[],\
                [],[],[],[],[],[]]

    resets = [[],[],[],[]]

    for i in range(len(letter_signal_paths)):
        spt = letter_signal_paths[i]
        ipt = letter_index_paths[i]
        signal = np.load(spt)
        index = np.load(ipt)

        letter_signal = []
        for idx in index:
            letter = signal[:, idx[0]:idx[1]]
            letter_signal.append(letter)

        for j in range(len(letter_signal)):
            letters[j].append(letter_signal[j])


    for i in range(len(reset_signal_paths)):
        spt = reset_signal_paths[i]
        ipt = reset_index_paths[i]
        signal = np.load(spt)
        index = np.load(ipt)

        reset_signal = []
        for idx in index:
            reset = signal[:, idx[0]:idx[1]]
            reset_signal.append(reset)

        for j in range(len(reset_signal)):
            resets[j].append(reset_signal[j])


    for i in range(len(letters)):
        letters[i].sort(key= lambda x:x.shape[1])
    
    for i in range(len(resets)):
        resets[i].sort(key= lambda x:x.shape[1])


    for rank in range(len(letters[0])):              # 列
        subcircle = []
        for j in range(len(letters)):                # 行
            subcircle.append(letters[j][rank])

        save_letter = subcircle[0]
        count = save_letter.shape[1]
        indexs = [np.array([0, count])]
        for i in range(1, len(subcircle)):
            letter = subcircle[i]
            indexs.append(np.array([count, count + letter.shape[1]]))
            count+=letter.shape[1]
            save_letter = np.concatenate((save_letter, letter), axis=1)
        
        indexs = np.array(indexs)

        np.save(os.path.join(save_path, "letter_rank_" + str(rank+1) + "signal.npy"), save_letter)
        np.save(os.path.join(save_path, "letter_rank_" + str(rank+1) + "index.npy"), indexs)
    
    
    for rank in range(len(resets[0])):              # 列
        subcircle = []
        for j in range(len(resets)):                # 行
            subcircle.append(resets[j][rank])
        
        save_letter = subcircle[0]
        count = save_letter.shape[1]
        indexs = [np.array([0, count])]
        for i in range(1, len(subcircle)):
            letter = subcircle[i]
            indexs.append(np.array([count, count + letter.shape[1]]))
            count+=letter.shape[1]
            save_letter = np.concatenate((save_letter, letter), axis=1)
        
        indexs = np.array(indexs)

        np.save(os.path.join(save_path, "reset_rank_" + str(rank+1) + "signal.npy"), save_letter)
        np.save(os.path.join(save_path, "reset_rank_" + str(rank+1) + "index.npy"), indexs)

# main3()

