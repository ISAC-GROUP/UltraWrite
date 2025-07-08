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

pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_2_0-CER=0.0039-WER=0.0141.ckpt")
# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_3_0-CER=0.0037-WER=0.0069.ckpt")
# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_5_0-CER=0.0037-WER=0.0107.ckpt")
model = pl_model.encoder
# model.cuda()
model.eval()


path = "../DATA/final_real_dataset_angle"

# ZPZ
test_n30_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
test_n30_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
test_n30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

test_n15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
test_n15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
test_n15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

test_p30_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
test_p30_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
test_p30_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

test_p15_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
test_p15_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
test_p15_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# LCZ
# test_n30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
# test_n30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
# test_n30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

# test_n15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
# test_n15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
# test_n15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

# test_p30_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
# test_p30_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
# test_p30_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

# test_p15_x = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
# test_p15_y = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
# test_p15_ids = np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()

# WDY
# test_n30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/datas.npy"), allow_pickle=True).tolist()
# test_n30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/labels.npy"), allow_pickle=True).tolist()
# test_n30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-30/ids.npy"), allow_pickle=True).tolist()

# test_n15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/datas.npy"), allow_pickle=True).tolist()
# test_n15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/labels.npy"), allow_pickle=True).tolist()
# test_n15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_-15/ids.npy"), allow_pickle=True).tolist()

# test_p30_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/datas.npy"), allow_pickle=True).tolist()
# test_p30_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/labels.npy"), allow_pickle=True).tolist()
# test_p30_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_30/ids.npy"), allow_pickle=True).tolist()

# test_p15_x = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/datas.npy"), allow_pickle=True).tolist()
# test_p15_y = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/labels.npy"), allow_pickle=True).tolist()
# test_p15_ids = np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_15/ids.npy"), allow_pickle=True).tolist()


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



def test(datas, labels, ids, uid, txtpath):
    encoder_outs = []
    label_outs = []
    id_outs = []

    test_x = datas
    test_y = labels
    test_ids = ids

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

    false_word = []
    for i in range(len(encoder_outs)):
        if(encoder_outs[i] != label_outs[i]):
            false_word.append(label_outs[i] + "-" + str(id_outs[i]))
    false_word = sorted(false_word)


    with open(os.path.join("/sdo/zcl/AcouWrite/Code", txtpath), "w") as f:
        for word in false_word:
            f.write(word + "\n")
    f.close()


    CER = torchmetrics.CharErrorRate()
    WER = torchmetrics.WordErrorRate()
    CER.reset()
    char_error_rate = CER(encoder_outs, label_outs)
    WER.reset()
    word_error_rate = WER(encoder_outs, label_outs)
    print("test user : " + str(uid), end=" ")
    print("CER : " + str(char_error_rate), end=" ")
    print("WER : " + str(word_error_rate), end=" ")
    print("WAcc : " + str(1 - word_error_rate), end=" ")
    print("\n")

# test(test_0_x, test_0_y, 2)

test(test_n30_x, test_n30_y, test_n30_ids, 2, "./check/ZPZ/zpz_-30.txt")
test(test_n15_x, test_n15_y, test_n15_ids, 2, "./check/ZPZ/zpz_-15.txt")
test(test_p30_x, test_p30_y, test_p30_ids, 2, "./check/ZPZ/zpz_30.txt")
test(test_p15_x, test_p15_y, test_p15_ids, 2, "./check/ZPZ/zpz_15.txt")

# test(test_n30_x, test_n30_y, test_n30_ids, 3, "./check/WDY/wdy_-30.txt")
# test(test_n15_x, test_n15_y, test_n15_ids, 3, "./check/WDY/wdy_-15.txt")
# test(test_p30_x, test_p30_y, test_p30_ids, 3, "./check/WDY/wdy_30.txt")
# test(test_p15_x, test_p15_y, test_p15_ids, 3, "./check/WDY/wdy_15.txt")

# test(test_n30_x, test_n30_y, test_n30_ids, 5, "./check/LCZ/lcz_-30.txt")
# test(test_n15_x, test_n15_y, test_n15_ids, 5, "./check/LCZ/lcz_-15.txt")
# test(test_p30_x, test_p30_y, test_p30_ids, 5, "./check/LCZ/lcz_30.txt")
# test(test_p15_x, test_p15_y, test_p15_ids, 5, "./check/LCZ/lcz_15.txt")
