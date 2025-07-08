import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import util
import TrainParam
import pytorch_lightning as pl
import PL_Model_v6
import PL_Model_4_3_2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

path = "../DATA/final_real_dataset"

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

    pl_model = PL_Model_4_3_2.pl_model(
            net_param=TrainParam.NET_PARAM, 
            train_param=TrainParam.TRAIN_PARAM,
            feature_extractor=feature_extractor,
            log_path=log_path
        )

    # 构建check point callback
    checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoint/paper_4_3_2_ckpt"),
            filename="4_3_2_bilstm_fenbu_" + uid + "_" + str(range) + "-{CER:.4f}-{WER:.4f}",
            monitor="CER",
            save_top_k=1,
            mode='min'
        )

    # 构建 logger
    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), "checkpoint/paper_4_3_2_logs"),
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


# fold_1
# train_x = []
# train_y = []
test_x = []
test_y = []

test_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
test_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10

# test_x = data_1
# test_y = label_1
# for i in range(1):
log_path="checkpoint/paper_4_3_2_TXT/4_3_2_bilstm_fenbu.txt"
train(train_x, train_y, test_x, test_y, "uid_1", 0, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
# train_y += label_1 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10

# test_x = data_2
# test_y = label_2
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_2_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_2", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
# train_y += label_1 + label_2 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10

# test_x = data_3
# test_y = label_3
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_3_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_3", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_5 + data_6 + data_7 + data_8 + data_9 + data_10
# train_y += label_1 + label_2 + label_3 + label_5 + label_6 + label_7 + label_8 + label_9 + label_10

# test_x = data_4
# test_y = label_4
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_4_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_4", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_6 + data_7 + data_8 + data_9 + data_10
# train_y += label_1 + label_2 + label_3 + label_4 + label_6 + label_7 + label_8 + label_9 + label_10

# test_x = data_5
# test_y = label_5
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_5_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_5", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_7 + data_8 + data_9 + data_10
# train_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_7 + label_8 + label_9 + label_10

# test_x = data_6
# test_y = label_6
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_6_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_6", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_8 + data_9 + data_10
# train_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_8 + label_9 + label_10

# test_x = data_7
# test_y = label_7
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_7_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_7", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_9 + data_10
# train_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_9 + label_10

# test_x = data_8
# test_y = label_8
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_8_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_8", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_10
# train_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_10

# test_x = data_9
# test_y = label_9
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_9_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_9", i, log_path)

# train_x = []
# train_y = []
# test_x = []
# test_y = []
# train_x += data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7 + data_8 + data_9
# train_y += label_1 + label_2 + label_3 + label_4 + label_5 + label_6 + label_7 + label_8 + label_9

# test_x = data_10
# test_y = label_10
# for i in range(1):
#     log_path="checkpoint/paper_4_3_2_TXT/4_3_2_birnn_uid_10_range_" + str(i) + ".txt"
#     train(train_x, train_y, test_x, test_y, "uid_10", i, log_path)
