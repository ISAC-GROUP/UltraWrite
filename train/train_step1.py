import numpy as np
import torch
from models import autoencoder
import torch.nn as nn

from torch.utils.data import DataLoader
from utils import util
import TrainParam
import os
import pytorch_lightning as pl
import PL_Model_v6
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything


train_x = np.load(os.path.join(TrainParam.TRAIN_DATASET_PATH, "synthe_test_datas.npy"), allow_pickle=True).tolist()
# test_x = np.load(os.path.join("../DATA/final_real_dataset/zcl_dataset_26c_25w_100ws_30st_v2", "datas.npy"), allow_pickle=True).tolist()
target_x = np.load(os.path.join("../DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2", "datas.npy"), allow_pickle=True).tolist()

datas = []
for x in train_x:
    for t in range(len(x)):
        datas.append(x[t])

# for x in test_x:
#     for t in range(len(x)):
#         datas.append(x[t])

datas = torch.tensor(datas, dtype=torch.float32)

labels = []
for x in target_x:
    for t in range(len(x)):
        labels.append(x[t])

labels = torch.tensor(labels, dtype=torch.float32)

train_dataset = util.MyDataset(datas, datas)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True, 
    num_workers=4
)

test_dataset = util.MyDataset(labels, labels)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True, 
    num_workers=4
)
seed_everything(2000)
# 构建 pytorch lightning model
pl_model = PL_Model_v6.pl_model(
        net_param=TrainParam.NET_PARAM, 
        train_param=TrainParam.TRAIN_PARAM
    )

# 构建check point callback
checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoint"),
        filename="encoder_autoencoder_vgg-{result_loss:.4f}",
        monitor="result_loss",
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
        max_epochs=50, 
        gpus=[5], 
        check_val_every_n_epoch=1
    )

# 开始训练
trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)