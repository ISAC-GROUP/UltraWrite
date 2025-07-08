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
import warnings
warnings.filterwarnings("ignore")


pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_1_0-CER=0.0009-WER=0.0031.ckpt")
model = pl_model.encoder
torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/mobilecode/code/model.pt")
# torch.save(model, "/sdo/zcl/AcouWrite/mobilecode/model/model.pt")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_2_0-CER=0.0039-WER=0.0141.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/zpzmodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_3_0-CER=0.0037-WER=0.0069.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/wdymodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_4_0-CER=0.0086-WER=0.0174.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/lmqmodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_5_0-CER=0.0037-WER=0.0107.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/lczmodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_6_0-CER=0.0032-WER=0.0057.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/cwymodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_7_0-CER=0.0097-WER=0.0292.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/lcfmodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_8_0-CER=0.0049-WER=0.0147.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/zymodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_9_0-CER=0.0219-WER=0.0653.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/cgjmodelseeword.pth")

# pl_model = PL_Model_4_3_4.pl_model.load_from_checkpoint("/sdo/zcl/AcouWrite/Code/checkpoint/paper_4_3_4_ckpt/4_3_4_cross_person_uid_10_0-CER=0.0132-WER=0.0388.ckpt")
# model = pl_model.encoder
# torch.save(model.state_dict(), "/sdo/zcl/AcouWrite/CorrectModel/model/gsmodelseeword.pth")
