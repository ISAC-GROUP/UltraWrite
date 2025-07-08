import numpy as np
import torch
import os
import matplotlib.pyplot as plt

_int2letter = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',\
    7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',\
    14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',\
    20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'
}

_int2reset = {0:"UD", 1:"MU", 2:"MD", 3:"DU"}

root_path = "../DATA/letter/ZCL/after_process_v12/"

signal_paths = []
index_paths = []

for root, dirs, files in os.walk(root_path):
    for file in files:
        pt = root + "/" + file
        if("signal" in file and "letter" in file):
            signal_paths.append(pt)
        elif("index" in file and "letter" in file):
            index_paths.append(pt)

signal_paths.sort()
index_paths.sort()


for i in range(len(signal_paths)):
    spt = signal_paths[i]
    ipt = index_paths[i]
    signal = np.load(spt)
    index = np.load(ipt)

    letter_signal = []
    for idx in index:
        letter = signal[:, idx[0]:idx[1]]
        letter_signal.append(letter)


    for j in range(len(letter_signal)):
        plt.figure(figsize=(6, 4))
        plt.title("LF4|" + str(_int2letter[j]))
        plt.imshow(np.flip(letter_signal[j], axis=0))
        plt.savefig("./img/ZCL/LF4/" + "rank_" + str(_int2letter[j]) + "_" + str(i+1) + ".png")
        plt.close()
    # exit(0)


