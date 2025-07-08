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
zpz_test_n30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

zpz_test_n15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
zpz_test_n15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
zpz_test_n15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

zpz_test_p30_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
zpz_test_p30_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
zpz_test_p30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

zpz_test_p15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
zpz_test_p15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
zpz_test_p15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# LCZ
lcz_test_0_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
lcz_test_0_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/lcz_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

lcz_test_n30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
lcz_test_n30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
lcz_test_n30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

lcz_test_n15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
lcz_test_n15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
lcz_test_n15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

lcz_test_p30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
lcz_test_p30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
lcz_test_p30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

lcz_test_p15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
lcz_test_p15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
lcz_test_p15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# WDY
wdy_test_0_x = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/datas.npy", allow_pickle=True).tolist()
wdy_test_0_y = np.load("/sdo/zcl/AcouWrite/DATA/final_real_dataset/wdy_dataset_26c_25w_100ws_30st_v2/labels.npy", allow_pickle=True).tolist()

wdy_test_n30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
wdy_test_n30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
wdy_test_n30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

wdy_test_n15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
wdy_test_n15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
wdy_test_n15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

wdy_test_p30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
wdy_test_p30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
wdy_test_p30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

wdy_test_p15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
wdy_test_p15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
wdy_test_p15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()


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


def test(model, datas, labels):
    encoder_outs = []
    label_outs = []
    id_outs = []

    test_x = datas
    test_y = labels

    test_dataset = util.MyDataset(test_x, test_y)
    # test_dataset = util.MyDataset_test(test_x, test_y, test_ids)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainParam.NUM_WORKERS,
        collate_fn=util.MyCollate
    )

    for step, (inputs, labels, inputs_lengths, labels_lengths) in enumerate(test_loader):
        inputs_lengths = inputs_lengths.cpu()
        # out = self(x, x_lengths)    # [b, ts, cln]
        encoder_outputs, encoder_hiddens = model(inputs, inputs_lengths)
        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for i in range(len(inputs_lengths)):
            mat = encoder_outputs[i]
            mat = mat[:inputs_lengths[i], :]
            encoder_outs.append(best_path(mat, CHARS))

        
        for label in labels:
            label = int2char(label)
            label_outs.append(label)

    return encoder_outs, label_outs

def getresult(correctWord, charlabels, txtpath):
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
        fp.write(str(char_error_rate) + " " + str(word_error_rate) + " " + str(1-word_error_rate) + "\n")
    fp.close()

txtpath = "./check/AngleNotCorrect.txt"
zpz_outs_0, zpz_labels_0 = test(zpz_model, zpz_test_0_x, zpz_test_0_y)
zpz_outs_n30, zpz_labels_n30 = test(zpz_model, zpz_test_n30_x, zpz_test_n30_y)
zpz_outs_n15, zpz_labels_n15 = test(zpz_model, zpz_test_n15_x, zpz_test_n15_y)
zpz_outs_p30, zpz_labels_p30 = test(zpz_model, zpz_test_p30_x, zpz_test_p30_y)
zpz_outs_p15, zpz_labels_p15 = test(zpz_model, zpz_test_p15_x, zpz_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()
print("zpz")
getresult(zpz_outs_0, zpz_labels_0,txtpath)
getresult(zpz_outs_n30, zpz_labels_n30,txtpath)
getresult(zpz_outs_n15, zpz_labels_n15,txtpath)
getresult(zpz_outs_p30, zpz_labels_p30,txtpath)
getresult(zpz_outs_p15, zpz_labels_p15,txtpath)


wdy_outs_0, wdy_labels_0 = test(wdy_model, wdy_test_0_x, wdy_test_0_y)
wdy_outs_n30, wdy_labels_n30 = test(wdy_model, wdy_test_n30_x, wdy_test_n30_y)
wdy_outs_n15, wdy_labels_n15 = test(wdy_model, wdy_test_n15_x, wdy_test_n15_y)
wdy_outs_p30, wdy_labels_p30 = test(wdy_model, wdy_test_p30_x, wdy_test_p30_y)
wdy_outs_p15, wdy_labels_p15 = test(wdy_model, wdy_test_p15_x, wdy_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()
print("wdy")
getresult(wdy_outs_0, wdy_labels_0, txtpath)
getresult(wdy_outs_n30, wdy_labels_n30, txtpath)
getresult(wdy_outs_n15, wdy_labels_n15, txtpath)
getresult(wdy_outs_p30, wdy_labels_p30, txtpath)
getresult(wdy_outs_p15, wdy_labels_p15, txtpath)


lcz_outs_0, lcz_labels_0 = test(lcz_model, lcz_test_0_x, lcz_test_0_y)
lcz_outs_n30, lcz_labels_n30 = test(lcz_model, lcz_test_n30_x, lcz_test_n30_y)
lcz_outs_n15, lcz_labels_n15 = test(lcz_model, lcz_test_n15_x, lcz_test_n15_y)
lcz_outs_p30, lcz_labels_p30 = test(lcz_model, lcz_test_p30_x, lcz_test_p30_y)
lcz_outs_p15, lcz_labels_p15 = test(lcz_model, lcz_test_p15_x, lcz_test_p15_y)
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()
print("lcz")
getresult(lcz_outs_0, lcz_labels_0, txtpath)
getresult(lcz_outs_n30, lcz_labels_n30, txtpath)
getresult(lcz_outs_n15, lcz_labels_n15, txtpath)
getresult(lcz_outs_p30, lcz_labels_p30, txtpath)
getresult(lcz_outs_p15, lcz_labels_p15, txtpath)

Total_outs_0 = zpz_outs_0 + wdy_outs_0 + lcz_outs_0
Total_labels_0 = zpz_labels_0 + wdy_labels_0 + lcz_labels_0

Total_outs_n30 = zpz_outs_n30 + wdy_outs_n30 + lcz_outs_n30
Total_labels_n30 = zpz_labels_n30 + wdy_labels_n30 + lcz_labels_n30

Total_outs_n15 = zpz_outs_n15 + wdy_outs_n15 + lcz_outs_n15
Total_labels_n15 = zpz_labels_n15 + wdy_labels_n15 + lcz_labels_n15

Total_outs_p30 = zpz_outs_p30 + wdy_outs_p30 + lcz_outs_p30
Total_labels_p30 = zpz_labels_p30 + wdy_labels_p30 + lcz_labels_p30

Total_outs_p15 = zpz_outs_p15 + wdy_outs_p15 + lcz_outs_p15
Total_labels_p15 = zpz_labels_p15 + wdy_labels_p15 + lcz_labels_p15
with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
fp.close()
print("Total")
getresult(Total_outs_0,Total_labels_0, txtpath)
getresult(Total_outs_n30, Total_labels_n30, txtpath)
getresult(Total_outs_n15, Total_labels_n15, txtpath)
getresult(Total_outs_p30, Total_labels_p30, txtpath)
getresult(Total_outs_p15, Total_labels_p15, txtpath)