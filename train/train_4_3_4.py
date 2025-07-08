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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

path = "../DATA/final_real_dataset"

# 读取train dataset
# 读取test dataset
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


def train(train_x, train_y, test_x, test_y, uid, range, log_path):
    # 创建 train loader 和 test loader
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

    model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_loss_v1_1-result_loss=0.0196.ckpt")
    feature_extractor = model.net.encoder

    # 构建 pytorch lightning model
    pl_model = PL_Model_4_3_4.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            feature_extractor=None,
            log_path=log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint/paper_4_3_4_ckpt"),
            filename="4_3_4_cross_person_merge_train_"+ uid + "_" + str(range) +"-{CER:.4f}-{WER:.4f}",
            monitor="CER",
            save_top_k=1,
            mode='min'
        )

    # 构建 logger
    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "checkpoint/paper_4_3_4_logs"), 
        )

    # 构建 trainer
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback], 
            logger=logger, 
            max_epochs=30, 
            gpus=TrainParam.GPUS, 
            check_val_every_n_epoch=1
        )

    # 开始训练
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_1_range_" + str(i) + ".txt"
    train(train_x, train_y, data_1 , label_1, "uid_1", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_2_range_" + str(i+5) + ".txt"
    train(train_x, train_y, data_2, label_2, "uid_2", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_3_range_" + str(i) + ".txt"
    train(train_x, train_y, data_3, label_3, "uid_3", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_4_range_" + str(i) + ".txt"
    train(train_x, train_y, data_4, label_4, "uid_4", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_5_range_" + str(i) + ".txt"
    train(train_x, train_y, data_5, label_5, "uid_5", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_6_range_" + str(i) + ".txt"
    train(train_x, train_y, data_6, label_6, "uid_6", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_7_range_" + str(i) + ".txt"
    train(train_x, train_y, data_7, label_7, "uid_7", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_8_range_" + str(i) + ".txt"
    train(train_x, train_y, data_8, label_8, "uid_8", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_9_range_" + str(i) + ".txt"
    train(train_x, train_y, data_9, label_9, "uid_9", i, log_path)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_merge_train_uid_10_range_" + str(i) + ".txt"
    train(train_x, train_y, data_10, label_10, "uid_10", i, log_path)