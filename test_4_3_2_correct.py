import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_2
import PL_Model_double_check
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  
import torch.nn.functional as F  
from ctc_decoder import *
import torchmetrics
import warnings
warnings.filterwarnings("ignore")

# rnn_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/rnn_model.pth")
# rnn_model.eval()
# exit(0)
# gru_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/gru_model.pth")
# gru_model.eval()

# lstm_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/lstm_model.pth")
# lstm_model.eval()

# birnn_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/birnn_model.pth")
# birnn_model.eval()

# bigru_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/bigru_model.pth")
# bigru_model.eval()

bilstm_model = torch.load("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_2_ckpt/bilstm_model.pth")
bilstm_model.eval()


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

# txtpath = "./check/test432/rnnCorrect.txt"
# with open(txtpath, "a") as fp:
#     fp.write("rnn" + "\n")
# fp.close()

# rnn_outs_top1, rnn_outs_top3, rnn_outs_top5, rnn_labels = test(rnn_model, test_x, test_y, top=5)
# print("rnn top 1")
# getresult(rnn_outs_top1, rnn_labels, txtpath, 1)
# print("rnn top 3")
# getresult(rnn_outs_top3, rnn_labels, txtpath, 3)
# print("rnn top 5")
# getresult(rnn_outs_top5, rnn_labels, txtpath, 5)


# txtpath = "./check/test432/gruCorrect.txt"
# with open(txtpath, "a") as fp:
#     fp.write("gru" + "\n")
# fp.close()

# gru_outs_top1, gru_outs_top3, gru_outs_top5, gru_labels = test(gru_model, test_x, test_y, top=5)
# print("gru top 1")
# getresult(gru_outs_top1, gru_labels, txtpath, 1)
# print("gru top 3")
# getresult(gru_outs_top3, gru_labels, txtpath, 3)
# print("gru top 5")
# getresult(gru_outs_top5, gru_labels, txtpath, 5)

# txtpath = "./check/test432/lstmCorrect.txt"
# with open(txtpath, "a") as fp:
#     fp.write("lstm" + "\n")
# fp.close()

# lstm_outs_top1, lstm_outs_top3, lstm_outs_top5, lstm_labels = test(lstm_model, test_x, test_y, top=5)
# print("lstm top 1")
# getresult(lstm_outs_top1, lstm_labels, txtpath, 1)
# print("lstm top 3")
# getresult(lstm_outs_top3, lstm_labels, txtpath, 3)
# print("lstm top 5")
# getresult(lstm_outs_top5, lstm_labels, txtpath, 5)


# txtpath = "./check/test432/birnnCorrect.txt"
# with open(txtpath, "a") as fp:
#     fp.write("birnn" + "\n")
# fp.close()

# birnn_outs_top1, birnn_outs_top3, birnn_outs_top5, birnn_labels = test(birnn_model, test_x, test_y, top=5)
# print("birnn top 1")
# getresult(birnn_outs_top1, birnn_labels, txtpath, 1)
# print("birnn top 3")
# getresult(birnn_outs_top3, birnn_labels, txtpath, 3)
# print("birnn top 5")
# getresult(birnn_outs_top5, birnn_labels, txtpath, 5)


# txtpath = "./check/test432/bigruCorrect.txt"
# with open(txtpath, "a") as fp:
#     fp.write("bigru" + "\n")
# fp.close()

# bigru_outs_top1, bigru_outs_top3, bigru_outs_top5, bigru_labels = test(bigru_model, test_x, test_y, top=5)
# print("bigru top 1")
# getresult(bigru_outs_top1, bigru_labels, txtpath, 1)
# print("bigru top 3")
# getresult(bigru_outs_top3, bigru_labels, txtpath, 3)
# print("bigru top 5")
# getresult(bigru_outs_top5, bigru_labels, txtpath, 5)

txtpath = "./check/test432/bilstmCorrect.txt"
with open(txtpath, "a") as fp:
    fp.write("bilstm" + "\n")
fp.close()

bilstm_outs_top1, bilstm_outs_top3, bilstm_outs_top5, bilstm_labels = test(bilstm_model, test_x, test_y, top=5)
print("bilstm top 1")
getresult(bilstm_outs_top1, bilstm_labels, txtpath, 1)
print("bilstm top 3")
getresult(bilstm_outs_top3, bilstm_labels, txtpath, 3)
print("bilstm top 5")
getresult(bilstm_outs_top5, bilstm_labels, txtpath, 5)