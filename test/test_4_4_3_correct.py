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


path = "../DATA/final_real_dataset_angle"

# ZPZ
zpz_test_0_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zpz_test_0_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

zpz_test_n30_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
zpz_test_n30_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
# zpz_test_n30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

zpz_test_n15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
zpz_test_n15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
# zpz_test_n15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

zpz_test_p30_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
zpz_test_p30_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
# zpz_test_p30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

zpz_test_p15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
zpz_test_p15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
# zpz_test_p15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# LCZ
lcz_test_0_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lcz_test_0_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()


lcz_test_n30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
lcz_test_n30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
# lcz_test_n30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

lcz_test_n15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
lcz_test_n15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
# lcz_test_n15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

lcz_test_p30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
lcz_test_p30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
# lcz_test_p30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

lcz_test_p15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
lcz_test_p15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
# lcz_test_p15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# WDY
wdy_test_0_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
wdy_test_0_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

wdy_test_n30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
wdy_test_n30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
# wdy_test_n30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

wdy_test_n15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
wdy_test_n15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
# wdy_test_n15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

wdy_test_p30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
wdy_test_p30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
# wdy_test_p30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

wdy_test_p15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
wdy_test_p15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
# wdy_test_p15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()


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

txtpath = "./check/test443/AngleCorrect.txt"


# test(test_0_x, test_0_y, 2)
zpz_0_correctWord_top1,zpz_0_correctWord_top3,zpz_0_correctWord_top5, zpz_0_charlabels = test(zpz_model, zpz_test_0_x, zpz_test_0_y)
zpz_n30_correctWord_top1,zpz_n30_correctWord_top3,zpz_n30_correctWord_top5, zpz_n30_charlabels = test(zpz_model, zpz_test_n30_x, zpz_test_n30_y)
zpz_n15_correctWord_top1,zpz_n15_correctWord_top3,zpz_n15_correctWord_top5, zpz_n15_charlabels = test(zpz_model, zpz_test_n15_x, zpz_test_n15_y)
zpz_p30_correctWord_top1,zpz_p30_correctWord_top3,zpz_p30_correctWord_top5, zpz_p30_charlabels = test(zpz_model, zpz_test_p30_x, zpz_test_p30_y)
zpz_p15_correctWord_top1,zpz_p15_correctWord_top3,zpz_p15_correctWord_top5, zpz_p15_charlabels = test(zpz_model, zpz_test_p15_x, zpz_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()
print("zpz")
getresult(zpz_0_correctWord_top1, zpz_0_charlabels,txtpath, top=1)
getresult(zpz_0_correctWord_top3, zpz_0_charlabels,txtpath, top=3)
getresult(zpz_0_correctWord_top5, zpz_0_charlabels,txtpath, top=5)

getresult(zpz_n30_correctWord_top1, zpz_n30_charlabels,txtpath, top=1)
getresult(zpz_n30_correctWord_top3, zpz_n30_charlabels,txtpath, top=3)
getresult(zpz_n30_correctWord_top5, zpz_n30_charlabels,txtpath, top=5)

getresult(zpz_n15_correctWord_top1, zpz_n15_charlabels,txtpath, top=1)
getresult(zpz_n15_correctWord_top3, zpz_n15_charlabels,txtpath, top=3)
getresult(zpz_n15_correctWord_top5, zpz_n15_charlabels,txtpath, top=5)

getresult(zpz_p30_correctWord_top1, zpz_p30_charlabels,txtpath, top=1)
getresult(zpz_p30_correctWord_top3, zpz_p30_charlabels,txtpath, top=3)
getresult(zpz_p30_correctWord_top5, zpz_p30_charlabels,txtpath, top=5)

getresult(zpz_p15_correctWord_top1, zpz_p15_charlabels,txtpath, top=1)
getresult(zpz_p15_correctWord_top3, zpz_p15_charlabels,txtpath, top=3)
getresult(zpz_p15_correctWord_top5, zpz_p15_charlabels,txtpath, top=5)

wdy_0_correctWord_top1,wdy_0_correctWord_top3,wdy_0_correctWord_top5, wdy_0_charlabels = test(wdy_model, wdy_test_0_x, wdy_test_0_y)
wdy_n30_correctWord_top1,wdy_n30_correctWord_top3,wdy_n30_correctWord_top5, wdy_n30_charlabels = test(wdy_model, wdy_test_n30_x, wdy_test_n30_y)
wdy_n15_correctWord_top1,wdy_n15_correctWord_top3,wdy_n15_correctWord_top5, wdy_n15_charlabels = test(wdy_model, wdy_test_n15_x, wdy_test_n15_y)
wdy_p30_correctWord_top1,wdy_p30_correctWord_top3,wdy_p30_correctWord_top5, wdy_p30_charlabels = test(wdy_model, wdy_test_p30_x, wdy_test_p30_y)
wdy_p15_correctWord_top1,wdy_p15_correctWord_top3,wdy_p15_correctWord_top5, wdy_p15_charlabels = test(wdy_model, wdy_test_p15_x, wdy_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()
print("wdy")
getresult(wdy_0_correctWord_top1, wdy_0_charlabels, txtpath, top=1)
getresult(wdy_0_correctWord_top3, wdy_0_charlabels, txtpath, top=3)
getresult(wdy_0_correctWord_top5, wdy_0_charlabels, txtpath, top=5)

getresult(wdy_n30_correctWord_top1, wdy_n30_charlabels, txtpath, top=1)
getresult(wdy_n30_correctWord_top3, wdy_n30_charlabels, txtpath, top=3)
getresult(wdy_n30_correctWord_top5, wdy_n30_charlabels, txtpath, top=5)

getresult(wdy_n15_correctWord_top1, wdy_n15_charlabels, txtpath, top=1)
getresult(wdy_n15_correctWord_top3, wdy_n15_charlabels, txtpath, top=3)
getresult(wdy_n15_correctWord_top5, wdy_n15_charlabels, txtpath, top=5)

getresult(wdy_p30_correctWord_top1, wdy_p30_charlabels, txtpath, top=1)
getresult(wdy_p30_correctWord_top3, wdy_p30_charlabels, txtpath, top=3)
getresult(wdy_p30_correctWord_top5, wdy_p30_charlabels, txtpath, top=5)

getresult(wdy_p15_correctWord_top1, wdy_p15_charlabels, txtpath, top=1)
getresult(wdy_p15_correctWord_top3, wdy_p15_charlabels, txtpath, top=3)
getresult(wdy_p15_correctWord_top5, wdy_p15_charlabels, txtpath, top=5)


lcz_0_correctWord_top1,lcz_0_correctWord_top3,lcz_0_correctWord_top5, lcz_0_charlabels = test(lcz_model, lcz_test_0_x, lcz_test_0_y)
lcz_n30_correctWord_top1, lcz_n30_correctWord_top3,lcz_n30_correctWord_top5, lcz_n30_charlabels = test(lcz_model, lcz_test_n30_x, lcz_test_n30_y)
lcz_n15_correctWord_top1, lcz_n15_correctWord_top3,lcz_n15_correctWord_top5, lcz_n15_charlabels = test(lcz_model, lcz_test_n15_x, lcz_test_n15_y)
lcz_p30_correctWord_top1, lcz_p30_correctWord_top3,lcz_p30_correctWord_top5, lcz_p30_charlabels = test(lcz_model, lcz_test_p30_x, lcz_test_p30_y)
lcz_p15_correctWord_top1, lcz_p15_correctWord_top3,lcz_p15_correctWord_top5, lcz_p15_charlabels = test(lcz_model, lcz_test_p15_x, lcz_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()
print("lcz")
getresult(lcz_0_correctWord_top1, lcz_0_charlabels, txtpath,top=1)
getresult(lcz_0_correctWord_top3, lcz_0_charlabels, txtpath,top=3)
getresult(lcz_0_correctWord_top5, lcz_0_charlabels, txtpath,top=5)

getresult(lcz_n30_correctWord_top1, lcz_n30_charlabels, txtpath,top=1)
getresult(lcz_n30_correctWord_top3, lcz_n30_charlabels, txtpath,top=3)
getresult(lcz_n30_correctWord_top5, lcz_n30_charlabels, txtpath,top=5)

getresult(lcz_n15_correctWord_top1, lcz_n15_charlabels, txtpath,top=1)
getresult(lcz_n15_correctWord_top3, lcz_n15_charlabels, txtpath,top=3)
getresult(lcz_n15_correctWord_top5, lcz_n15_charlabels, txtpath,top=5)

getresult(lcz_p30_correctWord_top1, lcz_p30_charlabels, txtpath,top=1)
getresult(lcz_p30_correctWord_top3, lcz_p30_charlabels, txtpath,top=3)
getresult(lcz_p30_correctWord_top5, lcz_p30_charlabels, txtpath,top=5)

getresult(lcz_p15_correctWord_top1, lcz_p15_charlabels, txtpath,top=1)
getresult(lcz_p15_correctWord_top3, lcz_p15_charlabels, txtpath,top=3)
getresult(lcz_p15_correctWord_top5, lcz_p15_charlabels, txtpath,top=5)

Total_0_correctWord_top1 = zpz_0_correctWord_top1 + wdy_0_correctWord_top1 + lcz_0_correctWord_top1
Total_0_correctWord_top3 = zpz_0_correctWord_top3 + wdy_0_correctWord_top3 + lcz_0_correctWord_top3
Total_0_correctWord_top5 = zpz_0_correctWord_top5 + wdy_0_correctWord_top5 + lcz_0_correctWord_top5
Total_0_charlabels = zpz_0_charlabels + wdy_0_charlabels + lcz_0_charlabels

Total_n30_correctWord_top1 = zpz_n30_correctWord_top1 + wdy_n30_correctWord_top1 + lcz_n30_correctWord_top1
Total_n30_correctWord_top3 = zpz_n30_correctWord_top3 + wdy_n30_correctWord_top3 + lcz_n30_correctWord_top3
Total_n30_correctWord_top5 = zpz_n30_correctWord_top5 + wdy_n30_correctWord_top5 + lcz_n30_correctWord_top5
Total_n30_charlabels = zpz_n30_charlabels + wdy_n30_charlabels + lcz_n30_charlabels

Total_n15_correctWord_top1 = zpz_n15_correctWord_top1 + wdy_n15_correctWord_top1 + lcz_n15_correctWord_top1
Total_n15_correctWord_top3 = zpz_n15_correctWord_top3 + wdy_n15_correctWord_top3 + lcz_n15_correctWord_top3
Total_n15_correctWord_top5 = zpz_n15_correctWord_top5 + wdy_n15_correctWord_top5 + lcz_n15_correctWord_top5
Total_n15_charlabels = zpz_n15_charlabels + wdy_n15_charlabels + lcz_n15_charlabels

Total_p30_correctWord_top1 = zpz_p30_correctWord_top1 + wdy_p30_correctWord_top1 + lcz_p30_correctWord_top1
Total_p30_correctWord_top3 = zpz_p30_correctWord_top3 + wdy_p30_correctWord_top3 + lcz_p30_correctWord_top3
Total_p30_correctWord_top5 = zpz_p30_correctWord_top5 + wdy_p30_correctWord_top5 + lcz_p30_correctWord_top5
Total_p30_charlabels = zpz_p30_charlabels + wdy_p30_charlabels + lcz_p30_charlabels

Total_p15_correctWord_top1 = zpz_p15_correctWord_top1 + wdy_p15_correctWord_top1 + lcz_p15_correctWord_top1
Total_p15_correctWord_top3 = zpz_p15_correctWord_top3 + wdy_p15_correctWord_top3 + lcz_p15_correctWord_top3
Total_p15_correctWord_top5 = zpz_p15_correctWord_top5 + wdy_p15_correctWord_top5 + lcz_p15_correctWord_top5
Total_p15_charlabels = zpz_p15_charlabels + wdy_p15_charlabels + lcz_p15_charlabels

with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
fp.close()
print("Total")
getresult(Total_0_correctWord_top1,Total_0_charlabels, txtpath,top=1)
getresult(Total_0_correctWord_top3,Total_0_charlabels, txtpath,top=3)
getresult(Total_0_correctWord_top5,Total_0_charlabels, txtpath,top=5)

getresult(Total_n30_correctWord_top1, Total_n30_charlabels, txtpath,top=1)
getresult(Total_n30_correctWord_top3, Total_n30_charlabels, txtpath,top=3)
getresult(Total_n30_correctWord_top5, Total_n30_charlabels, txtpath,top=5)

getresult(Total_n15_correctWord_top1, Total_n15_charlabels, txtpath,top=1)
getresult(Total_n15_correctWord_top3, Total_n15_charlabels, txtpath,top=3)
getresult(Total_n15_correctWord_top5, Total_n15_charlabels, txtpath,top=5)

getresult(Total_p30_correctWord_top1, Total_p30_charlabels, txtpath,top=1)
getresult(Total_p30_correctWord_top3, Total_p30_charlabels, txtpath,top=3)
getresult(Total_p30_correctWord_top5, Total_p30_charlabels, txtpath,top=5)

getresult(Total_p15_correctWord_top1, Total_p15_charlabels, txtpath,top=1)
getresult(Total_p15_correctWord_top3, Total_p15_charlabels, txtpath,top=3)
getresult(Total_p15_correctWord_top5, Total_p15_charlabels, txtpath,top=5)