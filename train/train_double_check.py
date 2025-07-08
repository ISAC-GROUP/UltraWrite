import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_double_check
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

path = "../DATA/final_real_dataset_angle"

test_x = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30_v2/datas.npy"), allow_pickle=True).tolist()
test_y = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30_v2/labels.npy"), allow_pickle=True).tolist()
test_ids = np.load(os.path.join(path, "zpz_dataset_26c_25w_100ws_30st_-30_v2/ids.npy")).tolist()



train_x = np.load(os.path.join(TrainParam.TRAIN_DATASET_PATH, "synthe_test_datas.npy"), allow_pickle=True).tolist()
train_y = np.load(os.path.join(TrainParam.TRAIN_DATASET_PATH, "synthe_test_labels.npy"), allow_pickle=True).tolist()
train_ids = [i for i in range(1, len(train_y)+1)]


def train(train_x, train_y, train_ids, test_x, test_y, test_ids, fold, log_path):
    # train_dataset = util.MyDataset(train_x, train_y)
    train_dataset = util.MyDataset_test(train_x, train_y, train_ids)
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=TrainParam.BATCH_SIZE,
    #     shuffle=True, 
    #     num_workers=TrainParam.NUM_WORKERS, 
    #     collate_fn=util.MyCollate
    # )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True, 
        num_workers=TrainParam.NUM_WORKERS, 
        collate_fn=util.MyCollate_test
    )

    # test_dataset = util.MyDataset(test_x, test_y)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=TrainParam.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=TrainParam.NUM_WORKERS,
    #     collate_fn=util.MyCollate
    # )
    
    test_dataset = util.MyDataset_test(test_x, test_y, test_ids)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainParam.NUM_WORKERS,
        collate_fn=util.MyCollate_test
    )
    seed_everything(2000)

    model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/encoder_autoencoder_loss_v1_1-result_loss=0.0196.ckpt")
    feature_extractor = model.net.encoder

    # 构建 pytorch lightning model
    pl_model = PL_Model_double_check.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            feature_extractor=feature_extractor,
            log_path = log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint"),
            filename="double_check_zpz_-30_seed_2000_v2_ts_" + fold + "-{CER:.4f}-{WER:.4f}",
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

def main():
    for i in range(1,2):   
        log_path = "checkpoint/TXT_Result/double_check_zpz_-30_seed_2000_v2_ts_fold_" + str(i) + ".txt"
        train(train_x, train_y, train_ids, test_x, test_y,test_ids, "fold_" + str(i), log_path)

main()