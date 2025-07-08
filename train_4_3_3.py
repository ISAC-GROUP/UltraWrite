import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_3
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
path = "../DATA/final_real_dataset"

datas = []
labels = []

datas += np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "zcl_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "wdy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "lmq_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "lmq_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "lcz_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "cwy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "cwy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "lcf_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "lcf_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "zy_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "zy_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "cgj_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "cgj_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()

datas += np.load(os.path.join(path, "gs_dataset_26c_25w_100ws_30st_v2/datas.npy"), allow_pickle=True).tolist()
labels += np.load(os.path.join(path, "gs_dataset_26c_25w_100ws_30st_v2/labels.npy"), allow_pickle=True).tolist()


train_x_1 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_1", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y_1 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_1", "synthe_test_labels.npy"), allow_pickle=True).tolist()

train_x_3 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_3", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y_3 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_3", "synthe_test_labels.npy"), allow_pickle=True).tolist()

train_x_5 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_5", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y_5 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_5", "synthe_test_labels.npy"), allow_pickle=True).tolist()

train_x_7 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_7", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y_7 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_7", "synthe_test_labels.npy"), allow_pickle=True).tolist()

train_x_9 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_9", "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y_9 = np.load(os.path.join("../DATA/synthe_dataset_reset_diff_speed_v9_9", "synthe_test_labels.npy"), allow_pickle=True).tolist()


def train(train_x, train_y, test_x, test_y, times, log_path):
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
    seed_everything(2023)

    model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_loss_v1_1-result_loss=0.0196.ckpt")
    feature_extractor = model.net.encoder

    pl_model = PL_Model_4_3_3.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            feature_extractor=feature_extractor,
            log_path=log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint/paper_4_3_3_ckpt"),
            filename="4_3_3_synthe_" + times + "-{CER:.4f}-{WER:.4f}",
            monitor="CER",
            save_top_k=1,
            mode='min'
        )

    # 构建 logger
    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "checkpoint/paper_4_3_3_logs"),
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



log_path = "checkpoint/paper_4_3_3_TXT/4_3_3_times_1.txt"
train(train_x_1, train_y_1, datas, labels, "times_1", log_path)

log_path = "checkpoint/paper_4_3_3_TXT/4_3_3_times_3.txt"
train(train_x_3, train_y_3, datas, labels, "times_3", log_path)

log_path = "checkpoint/paper_4_3_3_TXT/4_3_3_times_5.txt"
train(train_x_5, train_y_5, datas, labels, "times_5", log_path)

log_path = "checkpoint/paper_4_3_3_TXT/4_3_3_times_7.txt"
train(train_x_7, train_y_7, datas, labels, "times_7", log_path)

log_path = "checkpoint/paper_4_3_3_TXT/4_3_3_times_9.txt"
train(train_x_9, train_y_9, datas, labels, "times_9", log_path)