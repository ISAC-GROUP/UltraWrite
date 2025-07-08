import torchvision.models as tmodels
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import PL_Model_v6
import PL_Model_v7
import TrainParam
import os
import matplotlib.pyplot as plt
import warnings
from models import Seq2Seq

warnings.filterwarnings("ignore")

# 计算模型参数量


encoder = tmodels.mobilenet_v3_large()
encoder.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
encoder.classifier[3] = nn.Linear(1280, 128, bias=True)

# encoder = tmodels.vgg16()
# encoder.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# encoder.classifier[6] = nn.Linear(4096, 128, bias=True)

total = sum(p.numel() for p in encoder.parameters())

print("Total params: %.2fM" % (total/1e6))

