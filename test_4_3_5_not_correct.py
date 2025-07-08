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

zcl_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_1_0-CER=0.0980-WER=0.3034.ckpt")
zcl_model = zcl_pl_model.encoder
zcl_model.eval()

zpz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_2_0-CER=0.1058-WER=0.3063.ckpt")
zpz_model = zpz_pl_model.encoder
zpz_model.eval()

wdy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_3_0-CER=0.1101-WER=0.3024.ckpt")
wdy_model = wdy_pl_model.encoder
wdy_model.eval()

lmq_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_4_0-CER=0.1552-WER=0.4216.ckpt")
lmq_model = lmq_pl_model.encoder
lmq_model.eval()

lcz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_5_0-CER=0.1269-WER=0.3733.ckpt")
lcz_model = lcz_pl_model.encoder
lcz_model.eval()

cwy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_6_0-CER=0.1100-WER=0.3420.ckpt")
cwy_model = cwy_pl_model.encoder
cwy_model.eval()

lcf_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_7_0-CER=0.1142-WER=0.3450.ckpt")
lcf_model = lcf_pl_model.encoder
lcf_model.eval()

zy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_8_0-CER=0.1186-WER=0.3510.ckpt")
zy_model = zy_pl_model.encoder
zy_model.eval()

cgj_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_9_0-CER=0.1323-WER=0.3472.ckpt")
cgj_model = cgj_pl_model.encoder
cgj_model.eval()

gs_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_cross_person_and_unseen_word_uid_10_0-CER=0.1576-WER=0.4328.ckpt")
gs_model = gs_pl_model.encoder
gs_model.eval()

# zcl_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_1_0-CER=0.0642-WER=0.2167.ckpt")
# zcl_model = zcl_pl_model.encoder
# zcl_model.eval()

# zpz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_2_0-CER=0.0922-WER=0.2606.ckpt")
# zpz_model = zpz_pl_model.encoder
# zpz_model.eval()

# wdy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_3_0-CER=0.0951-WER=0.2990.ckpt")
# wdy_model = wdy_pl_model.encoder
# wdy_model.eval()

# lmq_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_4_0-CER=0.1149-WER=0.3554.ckpt")
# lmq_model = lmq_pl_model.encoder
# lmq_model.eval()

# lcz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_5_0-CER=0.0775-WER=0.2240.ckpt")
# lcz_model = lcz_pl_model.encoder
# lcz_model.eval()

# cwy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_6_0-CER=0.0775-WER=0.2443.ckpt")
# cwy_model = cwy_pl_model.encoder
# cwy_model.eval()

# lcf_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_7_0-CER=0.0931-WER=0.2661.ckpt")
# lcf_model = lcf_pl_model.encoder
# lcf_model.eval()

# zy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_8_0-CER=0.1007-WER=0.2773.ckpt")
# zy_model = zy_pl_model.encoder
# zy_model.eval()

# cgj_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_9_0-CER=0.1055-WER=0.2997.ckpt")
# cgj_model = cgj_pl_model.encoder
# cgj_model.eval()

# gs_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_5_ckpt/4_3_5_unseen_word_uid_10_0-CER=0.1312-WER=0.3493.ckpt")
# gs_model = gs_pl_model.encoder
# gs_model.eval()





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
        
        # out = best_path(mat, chars)
        # correctWord_top1.append(out)
        outs_top5 = beam_search(mat, chars, lm=None, lexicon=[], top=top) # top 5
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
    # return correctWord_top1, charlabels

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



zcldatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zcl_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zcllabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zcl_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/zclNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("zcl" + "\n")
fp.close()

zcl_outs_top1,zcl_outs_top3,zcl_outs_top5, zcl_labels = test(zcl_model, zcldatas, zcllabels, top=5)
print("zcl top 1")
getresult(zcl_outs_top1, zcl_labels, txtpath, 1)
print("zcl top 3")
getresult(zcl_outs_top3, zcl_labels, txtpath, 3)
print("zcl top 5")
getresult(zcl_outs_top5, zcl_labels, txtpath, 5)


zpzdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zpzlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/zpzNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()

zpz_outs_top1,zpz_outs_top3,zpz_outs_top5, zpz_labels = test(zpz_model, zpzdatas, zpzlabels, top=5)
print("zpz top 1")
getresult(zpz_outs_top1, zpz_labels, txtpath, 1)
print("zpz top 3")
getresult(zpz_outs_top3, zpz_labels, txtpath, 3)
print("zpz top 5")
getresult(zpz_outs_top5, zpz_labels, txtpath, 5)

wdydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
wdylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/wdyNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()

wdy_outs_top1,wdy_outs_top3,wdy_outs_top5, wdy_labels = test(wdy_model, wdydatas, wdylabels, top=5)
print("wdy top 1")
getresult(wdy_outs_top1, wdy_labels, txtpath, 1)
print("wdy top 3")
getresult(wdy_outs_top3, wdy_labels, txtpath, 3)
print("wdy top 5")
getresult(wdy_outs_top5, wdy_labels, txtpath, 5)

lmqdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lmq_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lmqlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lmq_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/lmqNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("lmq" + "\n")
fp.close()

lmq_outs_top1,lmq_outs_top3,lmq_outs_top5,lmq_labels = test(lmq_model, lmqdatas, lmqlabels, top=5)
print("lmq top 1")
getresult(lmq_outs_top1, lmq_labels, txtpath, 1)
print("lmq top 3")
getresult(lmq_outs_top3, lmq_labels, txtpath, 3)
print("lmq top 5")
getresult(lmq_outs_top5, lmq_labels, txtpath, 5)


lczdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lczlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/lczNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()

lcz_outs_top1,lcz_outs_top3,lcz_outs_top5, lcz_labels = test(lcz_model, lczdatas, lczlabels, top=5)
print("lcz top 1")
getresult(lcz_outs_top1, lcz_labels, txtpath, 1)
print("lcz top 3")
getresult(lcz_outs_top3, lcz_labels, txtpath, 3)
print("lcz top 5")
getresult(lcz_outs_top5, lcz_labels, txtpath, 5)


cwydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cwy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
cwylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cwy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/cwyNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("cwy" + "\n")
fp.close()

cwy_outs_top1,cwy_outs_top3,cwy_outs_top5, cwy_labels = test(cwy_model, cwydatas, cwylabels, top=5)
print("cwy top 1")
getresult(cwy_outs_top1, cwy_labels, txtpath, 1)
print("cwy top 3")
getresult(cwy_outs_top3, cwy_labels, txtpath, 3)
print("cwy top 5")
getresult(cwy_outs_top5, cwy_labels, txtpath, 5)


lcfdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcf_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lcflabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcf_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/lcfNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("lcf" + "\n")
fp.close()

lcf_outs_top1,lcf_outs_top3,lcf_outs_top5, lcf_labels = test(lcf_model, lcfdatas, lcflabels, top=5)
print("lcf top 1")
getresult(lcf_outs_top1, lcf_labels, txtpath, 1)
print("lcf top 3")
getresult(lcf_outs_top3, lcf_labels, txtpath, 3)
print("lcf top 5")
getresult(lcf_outs_top5, lcf_labels, txtpath, 5)

zydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/zyNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("zy" + "\n")
fp.close()

zy_outs_top1,zy_outs_top3,zy_outs_top5, zy_labels = test(zy_model, zydatas, zylabels, top=5)
print("zy top 1")
getresult(zy_outs_top1, zy_labels, txtpath, 1)
print("zy top 3")
getresult(zy_outs_top3, zy_labels, txtpath, 3)
print("zy top 5")
getresult(zy_outs_top5, zy_labels, txtpath, 5)

cgjdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cgj_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
cgjlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cgj_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/cgjNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("cgj" + "\n")
fp.close()

cgj_outs_top1,cgj_outs_top3,cgj_outs_top5, cgj_labels = test(cgj_model, cgjdatas, cgjlabels, top=5)
print("cgj top 1")
getresult(cgj_outs_top1, cgj_labels, txtpath, 1)
print("cgj top 3")
getresult(cgj_outs_top3, cgj_labels, txtpath, 3)
print("cgj top 5")
getresult(cgj_outs_top5, cgj_labels, txtpath, 5)

gsdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/gs_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
gslabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/gs_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

txtpath = "./check/test435/gsNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("gs" + "\n")
fp.close()

gs_outs_top1,gs_outs_top3,gs_outs_top5, gs_labels = test(gs_model, gsdatas, gslabels, top=5)
print("gs top 1")
getresult(gs_outs_top1, gs_labels, txtpath, 1)
print("gs top 3")
getresult(gs_outs_top3, gs_labels, txtpath, 3)
print("gs top 5")
getresult(gs_outs_top5, gs_labels, txtpath, 5)

Total_outs_top1 = zcl_outs_top1 + zpz_outs_top1 + wdy_outs_top1 + lmq_outs_top1 + lcz_outs_top1 + cwy_outs_top1 + lcf_outs_top1 + zy_outs_top1 + cgj_outs_top1 + gs_outs_top1
Total_outs_top3 = zcl_outs_top3 + zpz_outs_top3 + wdy_outs_top3 + lmq_outs_top3 + lcz_outs_top3 + cwy_outs_top3 + lcf_outs_top3 + zy_outs_top3 + cgj_outs_top3 + gs_outs_top3
Total_outs_top5 = zcl_outs_top5 + zpz_outs_top5 + wdy_outs_top5 + lmq_outs_top5 + lcz_outs_top5 + cwy_outs_top5 + lcf_outs_top5 + zy_outs_top5 + cgj_outs_top5 + gs_outs_top5
Total_labels = zcl_labels + zpz_labels + wdy_labels + lmq_labels + lcz_labels + cwy_labels + lcf_labels + zy_labels + cgj_labels + gs_labels
txtpath = "./check/test435/TotalNotCorrect_width1.txt"
with open(txtpath, "a") as fp:
    fp.write("Total" + "\n")
fp.close()
print("Total top 1")
getresult(Total_outs_top1, Total_labels, txtpath, 1)
print("Total top 3")
getresult(Total_outs_top3, Total_labels, txtpath, 3)
print("Total top 5")
getresult(Total_outs_top5, Total_labels, txtpath, 5)


print("avg samples : ", str(len(Total_labels)//10))
with open(txtpath, "a") as fp:
    fp.write("avg samples : " + str(len(Total_labels)//10))

fp.close()