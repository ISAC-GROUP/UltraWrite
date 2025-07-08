import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_v11
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 读取train dataset
# 读取test dataset
train_x = np.load(os.path.join(TrainParam.TRAIN_DATASET_PATH, "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y = np.load(os.path.join(TrainParam.TRAIN_DATASET_PATH, "synthe_test_labels.npy"), allow_pickle=True).tolist()
test_x = np.load(os.path.join(TrainParam.TEST_DATASET_PATH, "datas.npy"), allow_pickle=True).tolist()
test_y = np.load(os.path.join(TrainParam.TEST_DATASET_PATH, "labels.npy"), allow_pickle=True).tolist()


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


model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_loss_v1_1-result_loss=0.0196.ckpt")
feature_extractor = model.net.encoder

# 构建 pytorch lightning model
pl_model = PL_Model_v11.pl_model(
        net_param=TrainParam.NET_PARAM, 
        train_param=TrainParam.TRAIN_PARAM,
        feature_extractor=feature_extractor
    )

# 构建check point callback
checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoint"),
        filename="4_4_4_test_Xiaomi_-{CER:.4f}-{WER:.4f}",
        monitor="CER",
        save_top_k=1,
        mode='min'
    )

# 构建 logger
logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), TrainParam.LOG_PATH),
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