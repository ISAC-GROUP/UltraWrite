# for paper 4.3.1

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_1
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

path = "../DATA/final_real_dataset"
datas = []
labels = []
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

train_x = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9", "synthe_test_labels.npy"), allow_pickle=True).tolist()



def train(train_x, train_y, test_x, test_y, log_path, feature_extractor, flag):
    train_dataset = util.MyDataset(train_x, train_y)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True, 
        num_workers=TrainParam.NUM_WORKERS, 
        collate_fn=util.MyCollate
    )

    test_dataset = util.MyDataset(test_x, test_y)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainParam.NUM_WORKERS,
        collate_fn=util.MyCollate
    )
    seed_everything(2000)

    pl_model = PL_Model_4_3_1.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            feature_extractor=feature_extractor,
            log_path=log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint/paper_4_3_1_ckpt"),
            filename="4_3_1_" + flag + "_-{CER:.4f}-{WER:.4f}",
            monitor="CER",
            save_top_k=1,
            mode='min'
        )

    # 构建 logger
    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "checkpoint/paper_4_3_1_logs"),
        )

    # 构建 trainer
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback], 
            logger=logger, 
            max_epochs=50, 
            gpus=[4], 
            check_val_every_n_epoch=1
        )

    # 开始训练
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)



test_x = []
test_y = []

test_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
test_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10


model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_loss_v1_1-result_loss=0.0196.ckpt")
lenet5 = model.net.encoder

log_path="checkpoint/paper_4_3_1_TXT/4_3_1_lenet5_v2.txt"
train(train_x, train_y, test_x, test_y, log_path, lenet5, "lenet5")


# model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_Alexnet-result_loss=0.0184.ckpt")
# alexnet = model.net.encoder

# log_path="checkpoint/paper_4_3_1_TXT/4_3_1_alexnet.txt"
# train(train_x, train_y, test_x, test_y, log_path, alexnet, "alexnet")

# model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_mobilnet-result_loss=0.0129.ckpt")
# mobilnet = model.net.encoder

# log_path="checkpoint/paper_4_3_1_TXT/4_3_1_mobilnet.txt"
# train(train_x, train_y, test_x, test_y, log_path, mobilnet, "mobilnet")



# model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_resnet18-result_loss=0.0116.ckpt")
# resnet = model.net.encoder

# log_path="checkpoint/paper_4_3_1_TXT/4_3_1_resnet.txt"
# train(train_x, train_y, test_x, test_y, log_path, resnet, "resnet")

# model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_vgg-result_loss=0.0494.ckpt")
# vgg = model.net.encoder

# log_path="checkpoint/paper_4_3_1_TXT/4_3_1_vgg.txt"
# train(train_x, train_y, test_x, test_y, log_path, vgg, "vgg")