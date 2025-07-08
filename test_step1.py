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
import matplotlib.pyplot as plt
import torch.nn.functional as F

import PL_Model_4_3_4


target_x = np.load(os.path.join("../DATA/final_real_dataset/zpz_dataset_26c_25w_100ws_30st_v2", "datas.npy"), allow_pickle=True).tolist()


labels = []
for x in target_x:
    for t in range(len(x)):
        labels.append(x[t])

labels = torch.tensor(labels, dtype=torch.float32)



inputs = labels[6]
inputs = inputs.reshape(1,1,114,100)

f = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5f.npy")
t = np.load("/sdo/zcl/AcouWrite/DATA/forpaper/figure3-5t.npy")
t = t[:100]

model = PL_Model_v6.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/FeatureExtractor/lenet5_FeatureExtractor.ckpt")
encoder = model.net.encoder
decoder = model.net.decoder

outputs1 = decoder(encoder(inputs))
outputs1 = outputs1.detach().numpy()


pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_merge_train_uid_1_0-CER=0.0351-WER=0.0839.ckpt")
merge_encoder = pl_model.encoder.embed
merge_encoder.eval()

outputs = decoder(merge_encoder(inputs))
outputs = outputs.detach().numpy()


inputs = inputs.detach().numpy()
inputs = np.flip(inputs.reshape(114,100),axis=0)
outputs1 = np.flip(outputs1.reshape(114, 100),axis=0)
outputs = np.flip(outputs.reshape(114, 100), axis=0)

plt.figure(figsize=(8, 8))
plt.pcolormesh(t, f, inputs)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
# plt.colorbar()
plt.savefig("./img/testinput.png")
plt.close()


plt.figure(figsize=(8, 8))
plt.pcolormesh(t, f, outputs1)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
# plt.colorbar()
plt.savefig("./img/testoutput1.png")
plt.close()

plt.figure(figsize=(8, 8))
plt.pcolormesh(t, f, outputs)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
# plt.colorbar()
plt.savefig("./img/testoutput2.png")
plt.close()