import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_4_3_4_reconstruct
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

path = "../DATA/final_real_dataset"

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

def split_datas(datas, labels, test_idx):
    """
    将trg数据集切分为训练集和测试集
    每个单词手势出一个样本组成训练数据集
    """
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    for idx in range(len(labels)):
        data = datas[idx]
        label = labels[idx]
        if idx in test_idx:
            test_datas.append(data)
            test_labels.append(label)
        else:
            train_datas.append(data)
            train_labels.append(label)

    return train_datas, train_labels, test_datas, test_labels



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



def train(train_x_src, train_y_src, train_x_trg, train_y_trg, test_x, test_y, uid, range, log_path):

    train_loader = util.UnalignedDataLoader()
    train_loader.initialize(train_x_src, train_y_src, train_x_trg, train_y_trg)
    train_loader = train_loader.load_data()

    test_dataset = util.MyDataset(test_x, test_y)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TrainParam.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainParam.NUM_WORKERS,
        collate_fn=util.MyCollate
    )

    seed_everything(2000)

    # 构建 pytorch lightning model
    pl_model = PL_Model_4_3_4_reconstruct.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            log_path=log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint/paper_4_3_4_dfa_ckpt"),
            filename="4_3_4_cross_person_dfa_train_"+ uid + "_" + str(range) +"-{CER:.4f}-{WER:.4f}",
            monitor="CER",
            save_top_k=1,
            mode='min'
        )

    # 构建 logger
    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "checkpoint/paper_4_3_4_dfa_logs"), 
        )

    # 构建 trainer
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback], 
            logger=logger, 
            max_epochs=30, 
            gpus=[4], 
            check_val_every_n_epoch=1
        )

    # 开始训练
    trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_1_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user1_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_1,labels=label_1,test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_1", range=i,log_path=log_path)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_2_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user2_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_2,labels=label_2, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_2", range=i,log_path=log_path)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_3_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user3_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_3,labels=label_3, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_3", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_4_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user4_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_4,labels=label_4, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_4", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_5_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user5_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_5,labels=label_5, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_5", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_6_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user6_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_6,labels=label_6, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_6", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_7_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user7_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_7,labels=label_7,test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_7", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_8_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user8_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_8,labels=label_8, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_8", range=i,log_path=log_path)
# exit(0)

# for i in range(1):
#     log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_9_range_" + str(i) + ".txt"
#     test_idx = np.load(os.path.join(path, "user9_test_idx.npy"))
#     train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_9,labels=label_9, test_idx=test_idx)
#     train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_9", range=i,log_path=log_path)
# exit(0)

for i in range(1):
    log_path="checkpoint/paper_4_3_4_TXT/4_3_4_cross_person_dfa_train_uid_10_range_" + str(i) + ".txt"
    test_idx = np.load(os.path.join(path, "user10_test_idx.npy"))
    train_x_trg, train_y_trg, test_x, test_y = split_datas(datas=data_10,labels=label_10, test_idx=test_idx)
    train(train_x_src=train_x, train_y_src=train_y, train_x_trg=train_x_trg, train_y_trg=train_y_trg, test_x=test_x, test_y=test_y, uid="uid_10", range=i,log_path=log_path)
