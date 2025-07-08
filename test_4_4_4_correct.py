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


path = "../DATA/final_real_dataset_phone"

# ZPZ
zpz_test_Xiaomi_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_Xiaomi/datas.npy"), allow_pickle=True).tolist()
zpz_test_Xiaomi_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_Xiaomi/labels.npy"), allow_pickle=True).tolist()
zpz_test_Xiaomi_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_Xiaomi/ids.npy"), allow_pickle=True).tolist()

zpz_test_GalaxyS9_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_GalaxyS9/datas.npy"), allow_pickle=True).tolist()
zpz_test_GalaxyS9_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_GalaxyS9/labels.npy"), allow_pickle=True).tolist()
zpz_test_GalaxyS9_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_GalaxyS9/ids.npy"), allow_pickle=True).tolist()

# WDY
wdy_test_Xiaomi_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_Xiaomi/datas.npy"), allow_pickle=True).tolist()
wdy_test_Xiaomi_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_Xiaomi/labels.npy"), allow_pickle=True).tolist()
wdy_test_Xiaomi_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_Xiaomi/ids.npy"), allow_pickle=True).tolist()

wdy_test_GalaxyS9_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_GalaxyS9/datas.npy"), allow_pickle=True).tolist()
wdy_test_GalaxyS9_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_GalaxyS9/labels.npy"), allow_pickle=True).tolist()
wdy_test_GalaxyS9_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_GalaxyS9/ids.npy"), allow_pickle=True).tolist()

# LCZ
lcz_test_Xiaomi_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_Xiaomi/datas.npy"), allow_pickle=True).tolist()
lcz_test_Xiaomi_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_Xiaomi/labels.npy"), allow_pickle=True).tolist()
lcz_test_Xiaomi_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_Xiaomi/ids.npy"), allow_pickle=True).tolist()

lcz_test_GalaxyS9_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_GalaxyS9/datas.npy"), allow_pickle=True).tolist()
lcz_test_GalaxyS9_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_GalaxyS9/labels.npy"), allow_pickle=True).tolist()
lcz_test_GalaxyS9_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_GalaxyS9/ids.npy"), allow_pickle=True).tolist()


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


# def test(model, datas, labels, ids, uid, txtpath):
def test(model, datas, labels):
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
        # mat = F.log_softmax(mat, 2)
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

txtpath = "./check/test444/deviceCorrect.txt"

zpz_Xiaomi_correctWord_top1, zpz_Xiaomi_correctWord_top3, zpz_Xiaomi_correctWord_top5, zpz_Xiaomi_charlabels = test(zpz_model, zpz_test_Xiaomi_x, zpz_test_Xiaomi_y)
zpz_GalaxyS9_correctWord_top1, zpz_GalaxyS9_correctWord_top3, zpz_GalaxyS9_correctWord_top5, zpz_GalaxyS9_charlabels = test(zpz_model, zpz_test_GalaxyS9_x, zpz_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
    fp.write("Xiaomi" + "\n")
fp.close()

print("zpz")
getresult(zpz_Xiaomi_correctWord_top1, zpz_Xiaomi_charlabels, txtpath, top=1)
getresult(zpz_Xiaomi_correctWord_top3, zpz_Xiaomi_charlabels, txtpath,top=3)
getresult(zpz_Xiaomi_correctWord_top5, zpz_Xiaomi_charlabels, txtpath,top=5)

with open(txtpath, "a") as fp:
    fp.write("GalaxyS9"+"\n")
fp.close()
getresult(zpz_GalaxyS9_correctWord_top1, zpz_GalaxyS9_charlabels, txtpath, top=1)
getresult(zpz_GalaxyS9_correctWord_top3, zpz_GalaxyS9_charlabels, txtpath,top=3)
getresult(zpz_GalaxyS9_correctWord_top5, zpz_GalaxyS9_charlabels, txtpath,top=5)


wdy_Xiaomi_correctWord_top1, wdy_Xiaomi_correctWord_top3, wdy_Xiaomi_correctWord_top5, wdy_Xiaomi_charlabels = test(wdy_model, wdy_test_Xiaomi_x, wdy_test_Xiaomi_y)
wdy_GalaxyS9_correctWord_top1, wdy_GalaxyS9_correctWord_top3, wdy_GalaxyS9_correctWord_top5, wdy_GalaxyS9_charlabels = test(wdy_model, wdy_test_GalaxyS9_x, wdy_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
    fp.write("Xiaomi"+"\n")
fp.close()
print("wdy")
getresult(wdy_Xiaomi_correctWord_top1, wdy_Xiaomi_charlabels, txtpath, top=1)
getresult(wdy_Xiaomi_correctWord_top3, wdy_Xiaomi_charlabels, txtpath, top=3)
getresult(wdy_Xiaomi_correctWord_top5, wdy_Xiaomi_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("GalaxyS9"+"\n")
fp.close()
getresult(wdy_GalaxyS9_correctWord_top1, wdy_GalaxyS9_charlabels, txtpath, top=1)
getresult(wdy_GalaxyS9_correctWord_top3, wdy_GalaxyS9_charlabels, txtpath, top=3)
getresult(wdy_GalaxyS9_correctWord_top5, wdy_GalaxyS9_charlabels, txtpath, top=5)

lcz_Xiaomi_correctWord_top1,lcz_Xiaomi_correctWord_top3,lcz_Xiaomi_correctWord_top5, lcz_Xiaomi_charlabels = test(lcz_model, lcz_test_Xiaomi_x, lcz_test_Xiaomi_y)
lcz_GalaxyS9_correctWord_top1, lcz_GalaxyS9_correctWord_top3,lcz_GalaxyS9_correctWord_top5, lcz_GalaxyS9_charlabels = test(lcz_model, lcz_test_GalaxyS9_x, lcz_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
    fp.write("Xiaomi"+"\n")
fp.close()
print("lcz")
getresult(lcz_Xiaomi_correctWord_top1, lcz_Xiaomi_charlabels, txtpath, top=1)
getresult(lcz_Xiaomi_correctWord_top3, lcz_Xiaomi_charlabels, txtpath, top=3)
getresult(lcz_Xiaomi_correctWord_top5, lcz_Xiaomi_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("GalaxyS9"+ "\n")
fp.close()
getresult(lcz_GalaxyS9_correctWord_top1, lcz_GalaxyS9_charlabels, txtpath, top=1)
getresult(lcz_GalaxyS9_correctWord_top3, lcz_GalaxyS9_charlabels, txtpath, top=3)
getresult(lcz_GalaxyS9_correctWord_top5, lcz_GalaxyS9_charlabels, txtpath, top=5)

Total_Xiaomi_correctWord_top1 = zpz_Xiaomi_correctWord_top1 + wdy_Xiaomi_correctWord_top1 + lcz_Xiaomi_correctWord_top1
Total_Xiaomi_correctWord_top3 = zpz_Xiaomi_correctWord_top3 + wdy_Xiaomi_correctWord_top3 + lcz_Xiaomi_correctWord_top3
Total_Xiaomi_correctWord_top5 = zpz_Xiaomi_correctWord_top5 + wdy_Xiaomi_correctWord_top5 + lcz_Xiaomi_correctWord_top5
Total_Xiaomi_charlabels = zpz_Xiaomi_charlabels + wdy_Xiaomi_charlabels + lcz_Xiaomi_charlabels



Total_GalaxyS9_correctWord_top1 = zpz_GalaxyS9_correctWord_top1 + wdy_GalaxyS9_correctWord_top1 + lcz_GalaxyS9_correctWord_top1
Total_GalaxyS9_correctWord_top3 = zpz_GalaxyS9_correctWord_top3 + wdy_GalaxyS9_correctWord_top3 + lcz_GalaxyS9_correctWord_top3
Total_GalaxyS9_correctWord_top5 = zpz_GalaxyS9_correctWord_top5 + wdy_GalaxyS9_correctWord_top5 + lcz_GalaxyS9_correctWord_top5
Total_GalaxyS9_charlabels = zpz_GalaxyS9_charlabels + wdy_GalaxyS9_charlabels + lcz_GalaxyS9_charlabels

with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
    fp.write("Xiaomi"+"\n")
fp.close()
print("Total")
getresult(Total_Xiaomi_correctWord_top1, Total_Xiaomi_charlabels, txtpath,top=1)
getresult(Total_Xiaomi_correctWord_top3, Total_Xiaomi_charlabels, txtpath, top=3)
getresult(Total_Xiaomi_correctWord_top5, Total_Xiaomi_charlabels, txtpath, top=5)

with open(txtpath, "a") as fp:
    fp.write("GalaxyS9"+ "\n")
fp.close()
getresult(Total_GalaxyS9_correctWord_top1, Total_GalaxyS9_charlabels, txtpath,top=1)
getresult(Total_GalaxyS9_correctWord_top3, Total_GalaxyS9_charlabels, txtpath, top=3)
getresult(Total_GalaxyS9_correctWord_top5, Total_GalaxyS9_charlabels, txtpath, top=5)