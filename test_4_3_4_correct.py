import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_4, PL_Model_4_3_4_dann, PL_Model_4_3_4_reconstruct, PL_Model_4_3_4_dfa
import PL_Model_double_check
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  
import torch.nn.functional as F  
from ctc_decoder import *
import torchmetrics
import warnings
warnings.filterwarnings("ignore")

# zcl_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_1_0-CER=0.0009-WER=0.0031.ckpt")
# zcl_model = zcl_pl_model.encoder
# zcl_model.eval()

# zpz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_2_0-CER=0.0039-WER=0.0141.ckpt")
# zpz_model = zpz_pl_model.encoder
# zpz_model.eval()

# wdy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_3_0-CER=0.0037-WER=0.0069.ckpt")
# wdy_model = wdy_pl_model.encoder
# wdy_model.eval()

# lmq_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_4_0-CER=0.0086-WER=0.0174.ckpt")
# lmq_model = lmq_pl_model.encoder
# lmq_model.eval()

# lcz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_5_0-CER=0.0037-WER=0.0107.ckpt")
# lcz_model = lcz_pl_model.encoder
# lcz_model.eval()

# cwy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_6_0-CER=0.0032-WER=0.0057.ckpt")
# cwy_model = cwy_pl_model.encoder
# cwy_model.eval()

# lcf_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_7_0-CER=0.0097-WER=0.0292.ckpt")
# lcf_model = lcf_pl_model.encoder
# lcf_model.eval()

# zy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_8_0-CER=0.0049-WER=0.0147.ckpt")
# zy_model = zy_pl_model.encoder
# zy_model.eval()

# cgj_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_9_0-CER=0.0219-WER=0.0653.ckpt")
# cgj_model = cgj_pl_model.encoder
# cgj_model.eval()

# gs_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_10_0-CER=0.0132-WER=0.0388.ckpt")
# gs_model = gs_pl_model.encoder
# gs_model.eval()


# zcl_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_1_0-CER=0.0351-WER=0.0839.ckpt")
# zcl_model = zcl_pl_model.encoder
# zcl_model.eval()

# zpz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_2_0-CER=0.2942-WER=0.4613.ckpt")
# zpz_model = zpz_pl_model.encoder
# zpz_model.eval()

# wdy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_3_0-CER=0.2155-WER=0.3677.ckpt")
# wdy_model = wdy_pl_model.encoder
# wdy_model.eval()

# lmq_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_4_0-CER=0.2615-WER=0.3833.ckpt")
# lmq_model = lmq_pl_model.encoder
# lmq_model.eval()

# lcz_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_5_0-CER=0.2804-WER=0.4347.ckpt")
# lcz_model = lcz_pl_model.encoder
# lcz_model.eval()

# cwy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_6_0-CER=0.0902-WER=0.1897.ckpt")
# cwy_model = cwy_pl_model.encoder
# cwy_model.eval()

# lcf_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_7_0-CER=0.1830-WER=0.3421.ckpt")
# lcf_model = lcf_pl_model.encoder
# lcf_model.eval()

# zy_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_8_0-CER=0.1129-WER=0.2684.ckpt")
# zy_model = zy_pl_model.encoder
# zy_model.eval()

# cgj_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_9_0-CER=0.2151-WER=0.4036.ckpt")
# cgj_model = cgj_pl_model.encoder
# cgj_model.eval()

# gs_pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_10_0-CER=0.3144-WER=0.5015.ckpt")
# gs_model = gs_pl_model.encoder
# gs_model.eval()


# 对抗领域自适应
# zcl_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_1_0-CER=0.0216-WER=0.0408.ckpt")
# zcl_model = zcl_pl_model.net
# zcl_model.eval()

# zpz_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_2_0-CER=0.1545-WER=0.3014.ckpt")
# zpz_model = zpz_pl_model.net
# zpz_model.eval()

# wdy_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_3_0-CER=0.2433-WER=0.3957.ckpt")
# wdy_model = wdy_pl_model.net
# wdy_model.eval()

# lmq_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_4_0-CER=0.2283-WER=0.4006.ckpt")
# lmq_model = lmq_pl_model.net
# lmq_model.eval()

# lcz_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_5_0-CER=0.2399-WER=0.4227.ckpt")
# lcz_model = lcz_pl_model.net
# lcz_model.eval()

# cwy_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_6_0-CER=0.0563-WER=0.1238.ckpt")
# cwy_model = cwy_pl_model.net
# cwy_model.eval()

# lcf_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_7_0-CER=0.1372-WER=0.2871.ckpt")
# lcf_model = lcf_pl_model.net
# lcf_model.eval()

# zy_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_8_0-CER=0.0912-WER=0.2134.ckpt")
# zy_model = zy_pl_model.net
# zy_model.eval()

# cgj_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_9_0-CER=0.1547-WER=0.3099.ckpt")
# cgj_model = cgj_pl_model.net
# cgj_model.eval()

# gs_pl_model = PL_Model_4_3_4_dann.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dann_ckpt/4_3_4_cross_person_dann_train_uid_10_0-CER=0.1945-WER=0.3613.ckpt")
# gs_model = gs_pl_model.net
# gs_model.eval()

# 编码重构
# zcl_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_1_0-CER=0.0208-WER=0.0554.ckpt")
# zcl_model = zcl_pl_model.net
# zcl_model.eval()

# zpz_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_2_0-CER=0.1505-WER=0.2783.ckpt")
# zpz_model = zpz_pl_model.net
# zpz_model.eval()

# wdy_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_3_0-CER=0.1564-WER=0.3160.ckpt")
# wdy_model = wdy_pl_model.net
# wdy_model.eval()

# lmq_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_4_0-CER=0.1664-WER=0.3042.ckpt")
# lmq_model = lmq_pl_model.net
# lmq_model.eval()

# lcz_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_5_0-CER=0.1779-WER=0.3265.ckpt")
# lcz_model = lcz_pl_model.net
# lcz_model.eval()

# cwy_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_6_0-CER=0.0725-WER=0.1517.ckpt")
# cwy_model = cwy_pl_model.net
# cwy_model.eval()

# lcf_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_7_0-CER=0.1171-WER=0.2587.ckpt")
# lcf_model = lcf_pl_model.net
# lcf_model.eval()

# zy_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_8_0-CER=0.0886-WER=0.2389.ckpt")
# zy_model = zy_pl_model.net
# zy_model.eval()

# cgj_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_9_0-CER=0.1372-WER=0.2652.ckpt")
# cgj_model = cgj_pl_model.net
# cgj_model.eval()

# gs_pl_model = PL_Model_4_3_4_reconstruct.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_reconstruct_ckpt/4_3_4_cross_person_dann_train_uid_10_0-CER=0.1811-WER=0.3677.ckpt")
# gs_model = gs_pl_model.net
# gs_model.eval()

# 先验分布对齐
zcl_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_1_0-CER=0.0112-WER=0.0233.ckpt")
zcl_model = zcl_pl_model.net
zcl_model.eval()

zpz_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_2_0-CER=0.1465-WER=0.2841.ckpt")
zpz_model = zpz_pl_model.net
zpz_model.eval()

wdy_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_3_0-CER=0.2032-WER=0.3957.ckpt")
wdy_model = wdy_pl_model.net
wdy_model.eval()

lmq_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_4_0-CER=0.2032-WER=0.3253.ckpt")
lmq_model = lmq_pl_model.net
lmq_model.eval()

lcz_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_5_0-CER=0.1409-WER=0.2857.ckpt")
lcz_model = lcz_pl_model.net
lcz_model.eval()

cwy_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_6_0-CER=0.0639-WER=0.1424.ckpt")
cwy_model = cwy_pl_model.net
cwy_model.eval()

lcf_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_7_0-CER=0.1145-WER=0.2587.ckpt")
lcf_model = lcf_pl_model.net
lcf_model.eval()

zy_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_8_0-CER=0.0781-WER=0.2038.ckpt")
zy_model = zy_pl_model.net
zy_model.eval()

cgj_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_9_0-CER=0.1434-WER=0.2748.ckpt")
cgj_model = cgj_pl_model.net
cgj_model.eval()

gs_pl_model = PL_Model_4_3_4_dfa.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_dfa_ckpt/4_3_4_cross_person_dfa_train_uid_10_0-CER=0.1641-WER=0.3419.ckpt")
gs_model = gs_pl_model.net
gs_model.eval()

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

def split_datas(datas, labels, test_idx):
    """
    将trg数据集切分为训练集和测试集
    每个单词手势出一个样本组成训练数据集
    """
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    for idx in range(len(labels)):
        data = datas[idx]
        label = labels[idx]
        if idx in test_idx:
            test_datas.append(data)
            test_labels.append(label)
        else:
            train_datas.append(data)
            train_labels.append(label)

    return train_datas, train_labels, test_datas, test_labels


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
        # mat, _ = model(data, data_length)
        # mat, _, _ = model(data, data_length, 0) # dann
        mat, _, _, _ = model(data, data_length)
        mat = mat.cpu().detach().numpy()
        mat = mat[0]
        mat = softmax(mat)
        
        outs_top5 = beam_search(mat, chars, lm=lm, lexicon=lexicon, top=top) # top 5
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



zcldatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zcl_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zcllabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zcl_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
zcltestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user1_test_idx.npy")
_, _, zcldatas, zcllabels = split_datas(datas=zcldatas, labels=zcllabels, test_idx=zcltestidx)

txtpath = "./check/test434/zcldfaCorrects.txt"
# txtpath = "./check/test434/zclCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("zcl" + "\n")
fp.close()

zcl_outs_top1, zcl_outs_top3, zcl_outs_top5, zcl_labels = test(zcl_model, zcldatas, zcllabels, top=5)
print("zcl top 1")
getresult(zcl_outs_top1, zcl_labels, txtpath, 1)
print("zcl top 3")
getresult(zcl_outs_top3, zcl_labels, txtpath, 3)
print("zcl top 5")
getresult(zcl_outs_top5, zcl_labels, txtpath, 5)





zpzdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zpzlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
zpztestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user2_test_idx.npy")
_, _, zpzdatas, zpzlabels = split_datas(datas=zpzdatas, labels=zpzlabels, test_idx=zpztestidx)

txtpath = "./check/test434/zpzdfaCorrects.txt"
# txtpath = "./check/test434/zpzCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()

zpz_outs_top1, zpz_outs_top3, zpz_outs_top5, zpz_labels = test(zpz_model, zpzdatas, zpzlabels, top=5)
print("zpz top 1")
getresult(zpz_outs_top1, zpz_labels, txtpath, 1)
print("zpz top 3")
getresult(zpz_outs_top3, zpz_labels, txtpath, 3)
print("zpz top 5")
getresult(zpz_outs_top5, zpz_labels, txtpath, 5)




wdydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
wdylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
wdytestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user3_test_idx.npy")
_, _, wdydatas, wdylabels = split_datas(datas=wdydatas, labels=wdylabels, test_idx=wdytestidx)


txtpath = "./check/test434/wdydfaCorrects.txt"
# txtpath = "./check/test434/wdyCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()

wdy_outs_top1, wdy_outs_top3, wdy_outs_top5, wdy_labels = test(wdy_model, wdydatas, wdylabels, top=5)
print("wdy top 1")
getresult(wdy_outs_top1, wdy_labels, txtpath, 1)
print("wdy top 3")
getresult(wdy_outs_top3, wdy_labels, txtpath, 3)
print("wdy top 5")
getresult(wdy_outs_top5, wdy_labels, txtpath, 5)





lmqdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lmq_dataset_26c_25w_100ws_30st_bu/datas.npy", allow_pickle=True).tolist()
lmqlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lmq_dataset_26c_25w_100ws_30st_bu/labels.npy", allow_pickle=True).tolist()
lmqtestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user4_test_idx.npy")
_, _, lmqdatas, lmqlabels = split_datas(datas=lmqdatas, labels=lmqlabels, test_idx=lmqtestidx)

txtpath = "./check/test434/lmqdfaCorrects.txt"
# txtpath = "./check/test434/lmqCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("lmq" + "\n")
fp.close()

lmq_outs_top1, lmq_outs_top3, lmq_outs_top5, lmq_labels = test(lmq_model, lmqdatas, lmqlabels, top=5)
print("lmq top 1")
getresult(lmq_outs_top1, lmq_labels, txtpath, 1)
print("lmq top 3")
getresult(lmq_outs_top3, lmq_labels, txtpath, 3)
print("lmq top 5")
getresult(lmq_outs_top5, lmq_labels, txtpath, 5)




lczdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lczlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
lcztestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user5_test_idx.npy")
_, _, lczdatas, lczlabels = split_datas(datas=lczdatas, labels=lczlabels, test_idx=lcztestidx)
txtpath = "./check/test434/lczdfaCorrects.txt"
# txtpath = "./check/test434/lczCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()

lcz_outs_top1, lcz_outs_top3, lcz_outs_top5, lcz_labels = test(lcz_model, lczdatas, lczlabels, top=5)
print("lcz top 1")
getresult(lcz_outs_top1, lcz_labels, txtpath, 1)
print("lcz top 3")
getresult(lcz_outs_top3, lcz_labels, txtpath, 3)
print("lcz top 5")
getresult(lcz_outs_top5, lcz_labels, txtpath, 5)




cwydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cwy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
cwylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cwy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
cwytestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user6_test_idx.npy")
_, _, cwydatas, cwylabels = split_datas(datas=cwydatas, labels=cwylabels, test_idx=cwytestidx)
txtpath = "./check/test434/cwydfaCorrects.txt"
# txtpath = "./check/test434/cwyCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("cwy" + "\n")
fp.close()

cwy_outs_top1, cwy_outs_top3, cwy_outs_top5, cwy_labels = test(cwy_model, cwydatas, cwylabels, top=5)
print("cwy top 1")
getresult(cwy_outs_top1, cwy_labels, txtpath, 1)
print("cwy top 3")
getresult(cwy_outs_top3, cwy_labels, txtpath, 3)
print("cwy top 5")
getresult(cwy_outs_top5, cwy_labels, txtpath, 5)





lcfdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcf_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lcflabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcf_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
lcftestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user7_test_idx.npy")
_, _, lcfdatas, lcflabels = split_datas(datas=lcfdatas, labels=lcflabels, test_idx=lcftestidx)
txtpath = "./check/test434/lcfdfaCorrects.txt"
# txtpath = "./check/test434/lcfCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("lcf" + "\n")
fp.close()

lcf_outs_top1, lcf_outs_top3, lcf_outs_top5, lcf_labels = test(lcf_model, lcfdatas, lcflabels, top=5)
print("lcf top 1")
getresult(lcf_outs_top1, lcf_labels, txtpath, 1)
print("lcf top 3")
getresult(lcf_outs_top3, lcf_labels, txtpath, 3)
print("lcf top 5")
getresult(lcf_outs_top5, lcf_labels, txtpath, 5)



zydatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
zylabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/zy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
zytestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user8_test_idx.npy")
_, _, zydatas, zylabels = split_datas(datas=zydatas, labels=zylabels, test_idx=zytestidx)
txtpath = "./check/test434/zydfaCorrects.txt"
# txtpath = "./check/test434/zyCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("zy" + "\n")
fp.close()

zy_outs_top1, zy_outs_top3, zy_outs_top5, zy_labels = test(zy_model, zydatas, zylabels, top=5)
print("zy top 1")
getresult(zy_outs_top1, zy_labels, txtpath, 1)
print("zy top 3")
getresult(zy_outs_top3, zy_labels, txtpath, 3)
print("zy top 5")
getresult(zy_outs_top5, zy_labels, txtpath, 5)



cgjdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cgj_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
cgjlabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/cgj_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
cgjtestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user9_test_idx.npy")
_, _, cgjdatas, cgjlabels = split_datas(datas=cgjdatas, labels=cgjlabels, test_idx=cgjtestidx)
txtpath = "./check/test434/cgjdfaCorrects.txt"
# txtpath = "./check/test434/cgjCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("cgj" + "\n")
fp.close()

cgj_outs_top1, cgj_outs_top3, cgj_outs_top5, cgj_labels = test(cgj_model, cgjdatas, cgjlabels, top=5)
print("cgj top 1")
getresult(cgj_outs_top1, cgj_labels, txtpath, 1)
print("cgj top 3")
getresult(cgj_outs_top3, cgj_labels, txtpath, 3)
print("cgj top 5")
getresult(cgj_outs_top5, cgj_labels, txtpath, 5)




gsdatas = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/gs_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
gslabels = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/gs_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()
gstestidx = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/user10_test_idx.npy")
_, _, gsdatas, gslabels = split_datas(datas=gsdatas, labels=gslabels, test_idx=gstestidx)
txtpath = "./check/test434/gsdfaCorrects.txt"
# txtpath = "./check/test434/gsCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("gs" + "\n")
fp.close()

gs_outs_top1, gs_outs_top3, gs_outs_top5, gs_labels = test(gs_model, gsdatas, gslabels, top=5)
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

txtpath = "./check/test434/TotaldfaCorrects.txt"
# txtpath = "./check/test434/TotalCorrects.txt"
with open(txtpath, "a") as fp:
    fp.write("Total" + "\n")
fp.close()
print("Total top 1")
getresult(Total_outs_top1, Total_labels, txtpath, 1)
print("Total top 3")
getresult(Total_outs_top3, Total_labels, txtpath, 3)
print("Total top 5")
getresult(Total_outs_top5, Total_labels, txtpath, 5)