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

txtpath = "./check/DeviceNotCorrect.txt"


zpz_outs_Xiaomi, zpz_labels_Xiaomi = test(zpz_model, zpz_test_Xiaomi_x, zpz_test_Xiaomi_y)
zpz_outs_GalaxyS9, zpz_labels_GalaxyS9 = test(zpz_model, zpz_test_GalaxyS9_x, zpz_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("zpz" + "\n")
fp.close()

print("zpz")
getresult(zpz_outs_Xiaomi, zpz_labels_Xiaomi, txtpath)
getresult(zpz_outs_GalaxyS9, zpz_labels_GalaxyS9, txtpath)


wdy_outs_Xiaomi, wdy_labels_Xiaomi = test(wdy_model, wdy_test_Xiaomi_x, wdy_test_Xiaomi_y)
wdy_outs_GalaxyS9, wdy_labels_GalaxyS9 = test(wdy_model, wdy_test_GalaxyS9_x, wdy_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("wdy" + "\n")
fp.close()
print("wdy")
getresult(wdy_outs_Xiaomi, wdy_labels_Xiaomi, txtpath)
getresult(wdy_outs_GalaxyS9, wdy_labels_GalaxyS9, txtpath)

lcz_outs_Xiaomi, lcz_labels_Xiaomi = test(lcz_model, lcz_test_Xiaomi_x, lcz_test_Xiaomi_y)
lcz_outs_GalaxyS9, lcz_labels_GalaxyS9 = test(lcz_model, lcz_test_GalaxyS9_x, lcz_test_GalaxyS9_y)
with open(txtpath, "a") as fp:
    fp.write("lcz" + "\n")
fp.close()
print("lcz")
getresult(lcz_outs_Xiaomi, lcz_labels_Xiaomi, txtpath)
getresult(lcz_outs_GalaxyS9, lcz_labels_GalaxyS9, txtpath)


Total_outs_Xiaomi = zpz_outs_Xiaomi + wdy_outs_Xiaomi + lcz_outs_Xiaomi
Total_labels_Xiaomi = zpz_labels_Xiaomi + wdy_labels_Xiaomi + lcz_labels_Xiaomi
Total_outs_GalaxyS9 = zpz_outs_GalaxyS9 + wdy_outs_GalaxyS9 + lcz_outs_GalaxyS9
Total_labels_GalaxyS9 = zpz_labels_GalaxyS9 + wdy_labels_GalaxyS9 + lcz_labels_GalaxyS9

with open(txtpath, "a") as fp:
    fp.write("total" + "\n")
fp.close()
print("Total")
getresult(Total_outs_Xiaomi, Total_labels_Xiaomi, txtpath)
getresult(Total_outs_GalaxyS9, Total_labels_GalaxyS9, txtpath)