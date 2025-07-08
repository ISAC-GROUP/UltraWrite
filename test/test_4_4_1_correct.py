import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_4
import PL_Model_double_check
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  
import torch.nn.functional as F  
from ctc_decoder import *
import torchmetrics
import warnings
warnings.filterwarnings("ignore")

zpz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_2_0-CER=0.0039-WER=0.0141.ckpt")
wdy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_3_0-CER=0.0037-WER=0.0069.ckpt")
lcz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_5_0-CER=0.0037-WER=0.0107.ckpt")
zpz_model = zpz_pl_model.encoder
zpz_model.eval()
wdy_model = wdy_pl_model.encoder
wdy_model.eval()
lcz_model = lcz_pl_model.encoder
lcz_model.eval()


path = "../DATA/final_real_dataset_noise"

# ZPZ
zpz_test_60db_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_60db/datas.npy"), allow_pickle=True).tolist()
zpz_test_60db_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_60db/labels.npy"), allow_pickle=True).tolist()
zpz_test_60db_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_60db/ids.npy"), allow_pickle=True).tolist()

zpz_test_70db_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_70db/datas.npy"), allow_pickle=True).tolist()
zpz_test_70db_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_70db/labels.npy"), allow_pickle=True).tolist()
zpz_test_70db_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_70db/ids.npy"), allow_pickle=True).tolist()

# WDY
# test_50db_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_for_correct/datas.npy", allow_pickle=True).tolist()
# test_50db_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_for_correct/labels.npy", allow_pickle=True).tolist()
# test_50db_ids = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_for_correct/ids.npy", allow_pickle=True).tolist()

wdy_test_60db_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_60db_for_correct/datas.npy"), allow_pickle=True).tolist()
wdy_test_60db_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_60db_for_correct/labels.npy"), allow_pickle=True).tolist()
wdy_test_60db_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_60db_for_correct/ids.npy"), allow_pickle=True).tolist()

wdy_test_70db_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_70db/datas.npy"), allow_pickle=True).tolist()
wdy_test_70db_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_70db/labels.npy"), allow_pickle=True).tolist()
wdy_test_70db_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_70db/ids.npy"), allow_pickle=True).tolist()

# LCZ
lcz_test_60db_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_60db/datas.npy"), allow_pickle=True).tolist()
lcz_test_60db_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_60db/labels.npy"), allow_pickle=True).tolist()
lcz_test_60db_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_60db/ids.npy"), allow_pickle=True).tolist()

lcz_test_70db_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_70db/datas.npy"), allow_pickle=True).tolist()
lcz_test_70db_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_70db/labels.npy"), allow_pickle=True).tolist()
lcz_test_70db_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_70db/ids.npy"), allow_pickle=True).tolist()




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


def test(model, datas, labels, ids, uid, txtpath):
    correctWord_top1 = []
    correctWord_top3 = []
    correctWord_top5 = []
    charlabels = []
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

txtpath = "./check/test441/NoiseCorrect.txt"

zpz_60db_correctWord_top1, zpz_60db_correctWord_top3, zpz_60db_correctWord_top5, zpz_60db_charlabels = test(zpz_model, zpz_test_60db_x, zpz_test_60db_y, zpz_test_60db_ids, 2, "./check/ZPZ/zpz_60db_correct.txt")
zpz_70db_correctWord_top1, zpz_70db_correctWord_top3, zpz_70db_correctWord_top5, zpz_70db_charlabels = test(zpz_model, zpz_test_70db_x, zpz_test_70db_y, zpz_test_70db_ids, 2, "./check/ZPZ/zpz_70db_correct.txt")
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
    fp.write("60db" + "\n")
fp.close()
print("zpz")
getresult(zpz_60db_correctWord_top1, zpz_60db_charlabels, txtpath, top=1)
getresult(zpz_60db_correctWord_top3, zpz_60db_charlabels, txtpath, top=3)
getresult(zpz_60db_correctWord_top5, zpz_60db_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("70db" + "\n")
fp.close()
getresult(zpz_70db_correctWord_top1, zpz_70db_charlabels, txtpath, top=1)
getresult(zpz_70db_correctWord_top5, zpz_70db_charlabels, txtpath, top=3)
getresult(zpz_70db_correctWord_top5, zpz_70db_charlabels, txtpath, top=5)

wdy_60db_correctWord_top1,wdy_60db_correctWord_top3,wdy_60db_correctWord_top5, wdy_60db_charlabels = test(wdy_model, wdy_test_60db_x, wdy_test_60db_y, wdy_test_60db_ids, 3, "./check/WDY/wdy_60db_correct.txt")
wdy_70db_correctWord_top1,wdy_70db_correctWord_top3,wdy_70db_correctWord_top5, wdy_70db_charlabels = test(wdy_model, wdy_test_70db_x, wdy_test_70db_y, wdy_test_70db_ids, 3, "./check/WDY/wdy_70db_correct.txt")
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
    fp.write("60db" + "\n")
fp.close()

print("wdy")
getresult(wdy_60db_correctWord_top1, wdy_60db_charlabels, txtpath, top=1)
getresult(wdy_60db_correctWord_top3, wdy_60db_charlabels, txtpath, top=3)
getresult(wdy_60db_correctWord_top5, wdy_60db_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("70db" + "\n")
fp.close()
getresult(wdy_70db_correctWord_top1, wdy_70db_charlabels, txtpath, top=1)
getresult(wdy_70db_correctWord_top3, wdy_70db_charlabels, txtpath, top=3)
getresult(wdy_70db_correctWord_top5, wdy_70db_charlabels, txtpath, top=5)



lcz_60db_correctWord_top1,lcz_60db_correctWord_top3,lcz_60db_correctWord_top5, lcz_60db_charlabels = test(lcz_model, lcz_test_60db_x, lcz_test_60db_y, lcz_test_60db_ids, 5, "./check/LCZ/lcz_60db_correct.txt")
lcz_70db_correctWord_top1,lcz_70db_correctWord_top3,lcz_70db_correctWord_top5, lcz_70db_charlabels = test(lcz_model, lcz_test_70db_x, lcz_test_70db_y, lcz_test_70db_ids, 5, "./check/LCZ/lcz_70db_correct.txt")
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
    fp.write("60db" + "\n")
fp.close()
print("lcz")
getresult(lcz_60db_correctWord_top1, lcz_60db_charlabels, txtpath, top=1)
getresult(lcz_60db_correctWord_top3, lcz_60db_charlabels, txtpath, top=3)
getresult(lcz_60db_correctWord_top5, lcz_60db_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("70db" + "\n")
fp.close()
getresult(lcz_70db_correctWord_top1, lcz_70db_charlabels, txtpath, top=1)
getresult(lcz_70db_correctWord_top3, lcz_70db_charlabels, txtpath, top=3)
getresult(lcz_70db_correctWord_top5, lcz_70db_charlabels, txtpath, top=5)


TotalCorrectWord60db_top1 = zpz_60db_correctWord_top1 + wdy_60db_correctWord_top1 + lcz_60db_correctWord_top1
TotalCorrectWord60db_top3 = zpz_60db_correctWord_top3 + wdy_60db_correctWord_top3 + lcz_60db_correctWord_top3
TotalCorrectWord60db_top5 = zpz_60db_correctWord_top5 + wdy_60db_correctWord_top5 + lcz_60db_correctWord_top5
Totalcharlabels60db = zpz_60db_charlabels + wdy_60db_charlabels + lcz_60db_charlabels

TotalCorrectWord70db_top1 = zpz_70db_correctWord_top1 + wdy_70db_correctWord_top1 + lcz_70db_correctWord_top1
TotalCorrectWord70db_top3 = zpz_70db_correctWord_top3 + wdy_70db_correctWord_top3 + lcz_70db_correctWord_top3
TotalCorrectWord70db_top5 = zpz_70db_correctWord_top5 + wdy_70db_correctWord_top5 + lcz_70db_correctWord_top5
Totalcharlabels70db = zpz_70db_charlabels + wdy_70db_charlabels + lcz_70db_charlabels
with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
    fp.write("60db" + "\n")
fp.close()
print("Total")
getresult(TotalCorrectWord60db_top1, Totalcharlabels60db, txtpath, top=1)
getresult(TotalCorrectWord60db_top3, Totalcharlabels60db, txtpath, top=3)
getresult(TotalCorrectWord60db_top5, Totalcharlabels60db, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("70db" + "\n")
fp.close()
getresult(TotalCorrectWord70db_top1, Totalcharlabels70db, txtpath, top=1)
getresult(TotalCorrectWord70db_top3, Totalcharlabels70db, txtpath, top=3)
getresult(TotalCorrectWord70db_top5, Totalcharlabels70db, txtpath, top=5)
