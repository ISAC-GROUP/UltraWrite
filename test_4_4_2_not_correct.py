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

path = "../DATA/final_real_dataset_distance"

# ZPZ
# test_5cm_x = np.load(os.path.join("../DATA/final_real_dataset", "lcz_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
# test_5cm_y = np.load(os.path.join("../DATA/final_real_dataset", "lcz_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

zpz_test_15cm_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15cm/datas.npy"), allow_pickle=True).tolist()
zpz_test_15cm_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15cm/labels.npy"), allow_pickle=True).tolist()
zpz_test_15cm_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15cm/ids.npy"), allow_pickle=True).tolist()

zpz_test_25cm_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_25cm/datas.npy"), allow_pickle=True).tolist()
zpz_test_25cm_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_25cm/labels.npy"), allow_pickle=True).tolist()
zpz_test_25cm_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_25cm/ids.npy"), allow_pickle=True).tolist()

# WDY
wdy_test_15cm_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15cm/datas.npy"), allow_pickle=True).tolist()
wdy_test_15cm_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15cm/labels.npy"), allow_pickle=True).tolist()
wdy_test_15cm_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15cm/ids.npy"), allow_pickle=True).tolist()

wdy_test_25cm_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_25cm/datas.npy"), allow_pickle=True).tolist()
wdy_test_25cm_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_25cm/labels.npy"), allow_pickle=True).tolist()
wdy_test_25cm_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_25cm/ids.npy"), allow_pickle=True).tolist()

# LCZ
lcz_test_15cm_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15cm/datas.npy"), allow_pickle=True).tolist()
lcz_test_15cm_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15cm/labels.npy"), allow_pickle=True).tolist()
lcz_test_15cm_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15cm/ids.npy"), allow_pickle=True).tolist()

lcz_test_25cm_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_25cm/datas.npy"), allow_pickle=True).tolist()
lcz_test_25cm_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_25cm/labels.npy"), allow_pickle=True).tolist()
lcz_test_25cm_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_25cm/ids.npy"), allow_pickle=True).tolist()


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


def test(model, datas, labels, ids, uid, txtpath):
    encoder_outs = []
    label_outs = []
    id_outs = []

    test_x = datas
    test_y = labels
    test_ids = ids

    # test_dataset = util.MyDataset(test_x, test_y)
    test_dataset = util.MyDataset_test(test_x, test_y, test_ids)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainParam.NUM_WORKERS,
        collate_fn=util.MyCollate_test
    )

    for step, (inputs, labels, inputs_lengths, labels_lengths, ids_step) in enumerate(test_loader):
        inputs_lengths = inputs_lengths.cpu()
        # out = self(x, x_lengths)    # [b, ts, cln]
        encoder_outputs, encoder_hiddens = model(inputs, inputs_lengths)
        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        
        # for mat in encoder_outputs:
        #     encoder_outs.append(best_path(mat, self.train_param["CHARS"]))
        for i in range(len(inputs_lengths)):
            mat = encoder_outputs[i]
            mat = mat[:inputs_lengths[i], :]
            encoder_outs.append(best_path(mat, CHARS))

        
        for label in labels:
            label = int2char(label)
            label_outs.append(label)

        id_outs += ids_step.tolist()

    # false_word = []
    # for i in range(len(encoder_outs)):
    #     if(encoder_outs[i] != label_outs[i]):
    #         false_word.append(label_outs[i] + "-" + str(id_outs[i]))
    # false_word = sorted(false_word)


    # with open(os.path.join("/sdo/zcl/AcouWrite/Code", txtpath), "w") as f:
    #     for word in false_word:
    #         f.write(word + "\n")
    # f.close()

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

txtpath = "./check/DistanceNotCorrect.txt"

zpz_outs_15cm, zpz_labels_15cm = test(zpz_model, zpz_test_15cm_x, zpz_test_15cm_y, zpz_test_15cm_ids, 2, "./check/ZPZ/zpz_15cm.txt")
zpz_outs_25cm, zpz_labels_25cm = test(zpz_model, zpz_test_25cm_x, zpz_test_25cm_y, zpz_test_25cm_ids, 2, "./check/ZPZ/zpz_25cm.txt")
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()

print("zpz")
getresult(zpz_outs_15cm, zpz_labels_15cm, txtpath)
getresult(zpz_outs_25cm, zpz_labels_25cm, txtpath)


wdy_outs_15cm, wdy_labels_15cm = test(wdy_model, wdy_test_15cm_x, wdy_test_15cm_y, wdy_test_15cm_ids, 3, "./check/WDY/wdy_15cm.txt")
wdy_outs_25cm, wdy_labels_25cm = test(wdy_model, wdy_test_25cm_x, wdy_test_25cm_y, wdy_test_25cm_ids, 3, "./check/WDY/wdy_25cm.txt")
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()

print("wdy")
getresult(wdy_outs_15cm, wdy_labels_15cm, txtpath)
getresult(wdy_outs_25cm, wdy_labels_25cm, txtpath)


lcz_outs_15cm, lcz_labels_15cm = test(lcz_model, lcz_test_15cm_x, lcz_test_15cm_y, lcz_test_15cm_ids, 5, "./check/LCZ/lcz_15cm.txt")
lcz_outs_25cm, lcz_labels_25cm = test(lcz_model, lcz_test_25cm_x, lcz_test_25cm_y, lcz_test_25cm_ids, 5, "./check/LCZ/lcz_25cm.txt")

with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()
print("lcz")
getresult(lcz_outs_15cm, lcz_labels_15cm, txtpath)
getresult(lcz_outs_25cm, lcz_labels_25cm, txtpath)


TotalOuts15cm = zpz_outs_15cm + wdy_outs_15cm + lcz_outs_15cm
TotalLabels15cm = zpz_labels_15cm + wdy_labels_15cm + lcz_labels_15cm
TotalOuts25cm = zpz_outs_25cm + wdy_outs_25cm + lcz_outs_25cm
TotalLabels25cm = zpz_labels_25cm + wdy_labels_25cm + lcz_labels_25cm
with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
fp.close()
print("Total")
getresult(TotalOuts15cm, TotalLabels15cm, txtpath)
getresult(TotalOuts25cm, TotalLabels25cm, txtpath)