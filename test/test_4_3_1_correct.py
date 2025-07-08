import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_1
import PL_Model_double_check
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  
import torch.nn.functional as F  
from ctc_decoder import *
import torchmetrics
import warnings
warnings.filterwarnings("ignore")

lenet5_pl_model = PL_Model_4_3_1.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_1_ckpt/4_3_1_lenet5_-CER=0.0075-WER=0.0205.ckpt")
lenet5_model = lenet5_pl_model.net
lenet5_model.eval()

alexnet_pl_model = PL_Model_4_3_1.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_1_ckpt/4_3_1_alexnet_-CER=0.0363-WER=0.1009.ckpt")
alexnet_model = alexnet_pl_model.net
alexnet_model.eval()

mobilnet_pl_model = PL_Model_4_3_1.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_1_ckpt/4_3_1_mobilnet_-CER=0.0363-WER=0.0960.ckpt")
mobilnet_model = mobilnet_pl_model.net
mobilnet_model.eval()

resnet_pl_model = PL_Model_4_3_1.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_1_ckpt/4_3_1_resnet_-CER=0.0702-WER=0.1469.ckpt")
resnet_model = resnet_pl_model.net
resnet_model.eval()


path = "../DATA/final_real_dataset"

data_1 = np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_1 = np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_2 = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_2 = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_3 = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_3 = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_4 = np.load(os.path.join(path, "lmq_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_4 = np.load(os.path.join(path, "lmq_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_5 = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_5 = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_6 = np.load(os.path.join(path, "cwy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_6 = np.load(os.path.join(path, "cwy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_7 = np.load(os.path.join(path, "lcf_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_7 = np.load(os.path.join(path, "lcf_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_8 = np.load(os.path.join(path, "zy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_8 = np.load(os.path.join(path, "zy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_9 = np.load(os.path.join(path, "cgj_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_9 = np.load(os.path.join(path, "cgj_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

data_10 = np.load(os.path.join(path, "gs_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
label_10 = np.load(os.path.join(path, "gs_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

test_x = []
test_y = []

test_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
test_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10



CHARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z']

def int2char(label) -> str:
    result = ""
    for l in label:
        if l >= len(CHARS):       # 当前值为padding的
            continue
        else:
            result += CHARS[l]  # 当前值是正常值
    
    return result


def corpus():
    with open('./resource/COCA.txt') as f:
        txt = f.read()
    return txt

def softmax(mat):
    maxT, _ = mat.shape  # dim0=t, dim1=c
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s
    return res


chars = "abcdefghijklmnopqrstuvwxyz"
corpuss = corpus()


lexicon = []
with open("./resource/COCA.txt", 'r') as fp:
    for line in fp:
        lexicon.append(line.strip("\n"))

fp.close()

lm = LanguageModel(corpuss, chars)


def test(model, datas, labels, top):
    correctWord_top1 = []
    correctWord_top3 = []
    correctWord_top5 = []
    charlabels = []
    # false_word = []
    for label in labels:
        charlabels.append(int2char(label).lower())

    for i in range(len(labels)):
        data = datas[i]
        data = torch.from_numpy(data).float()
        data = torch.unsqueeze(data, 0)
        data_length = torch.tensor([data.shape[1]], dtype=torch.int64)
        mat, _ = model(data, data_length)
        mat = mat.cpu().detach().numpy()
        mat = mat[0]
        mat = softmax(mat)
        
        outs_top5 = beam_search(mat, chars, lm=lm, lexicon=lexicon, top=5) # top 5
        outs_top3 = outs_top5[:3]
        outs_top1 = outs_top5[0]
        # print(outs)
        label = charlabels[i]
        if label in outs_top5:
            correctWord_top5.append(label)
        else:
            correctWord_top5.append(outs_top5[0])

        if label in outs_top3:
            correctWord_top3.append(label)
        else:
            correctWord_top3.append(outs_top3[0])

        if label == outs_top1:
            correctWord_top1.append(label)
        else:
            correctWord_top1.append(outs_top5[0])


    return correctWord_top1, correctWord_top3, correctWord_top5, charlabels

def getresult(correctWord, charlabels, txtpath, top):
    CER = torchmetrics.CharErrorRate()
    WER = torchmetrics.WordErrorRate()
    CER.reset()
    char_error_rate = CER(correctWord, charlabels)
    WER.reset()
    word_error_rate = WER(correctWord, charlabels)
    print("CER : " + str(char_error_rate), end=" ")
    print("WER : " + str(word_error_rate), end=" ")
    print("WAcc : " + str(1 - word_error_rate), end=" ")
    print("\n")
    with open(txtpath, "a") as fp:
        fp.write(str(top) + "\n")
        fp.write(str(char_error_rate) + " " + str(word_error_rate) + " " + str(1-word_error_rate) + "\n")
    fp.close()

txtpath = "./check/test431/lenet5Correct.txt"

with open(txtpath, "a") as fp:
    fp.write("lenet5" + "\n")
fp.close()

lenet5_outs_top1, lenet5_outs_top3, lenet5_outs_top5, lenet5_labels = test(lenet5_model, test_x, test_y, top=5)
print("lenet5 top 1")
getresult(lenet5_outs_top1, lenet5_labels, txtpath, 1)
print("lenet5 top 3")
getresult(lenet5_outs_top3, lenet5_labels, txtpath, 3)
print("lenet5 top 5")
getresult(lenet5_outs_top5, lenet5_labels, txtpath, 5)

txtpath = "./check/test431/alexnetCorrect.txt"
with open(txtpath, "a") as fp:
    fp.write("alexnet" + "\n")
fp.close()

alexnet_outs_top1, alexnet_outs_top3, alexnet_outs_top5, alexnet_labels = test(alexnet_model, test_x, test_y, top=5)
print("alexnet top 1")
getresult(alexnet_outs_top1, alexnet_labels, txtpath, 1)
print("alexnet top 3")
getresult(alexnet_outs_top3, alexnet_labels, txtpath, 3)
print("alexnet top 5")
getresult(alexnet_outs_top5, alexnet_labels, txtpath, 5)

txtpath = "./check/test431/mobilnetCorrect.txt"
with open(txtpath, "a") as fp:
    fp.write("mobilnet" + "\n")
fp.close()

mobilnet_outs_top1, mobilnet_outs_top3, mobilnet_outs_top5, mobilnet_labels = test(mobilnet_model, test_x, test_y, top=5)
print("mobilnet top 1")
getresult(mobilnet_outs_top1, mobilnet_labels, txtpath, 1)
print("mobilnet top 3")
getresult(mobilnet_outs_top3, mobilnet_labels, txtpath, 3)
print("mobilnet top 5")
getresult(mobilnet_outs_top5, mobilnet_labels, txtpath, 5)

txtpath = "./check/test431/resnetCorrect.txt"
with open(txtpath, "a") as fp:
    fp.write("resnet" + "\n")
fp.close()

resnet_outs_top1, resnet_outs_top3, resnet_outs_top5, resnet_labels = test(resnet_model, test_x, test_y, top=5)
print("resnet top 1")
getresult(resnet_outs_top1, resnet_labels, txtpath, 1)
print("resnet top 3")
getresult(resnet_outs_top3, resnet_labels, txtpath, 3)
print("resnet top 5")
getresult(resnet_outs_top5, resnet_labels, txtpath, 5)