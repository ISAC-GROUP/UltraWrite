# ========================================== #
#  this version is a seq2seq framework model #
# ========================================== #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from models import autoencoder
import torchmetrics
from ctc_decoder import *

class pl_model(pl.LightningModule):
    def __init__(self, net_param, train_param):
        super(pl_model, self).__init__()
        """
        pytorch lightning module
        
        构造函数需要完成任务：
            设置训练模型，参数来自于net param

            设置CTC损失

            设置metric 此次使用的指标是【Word error rate】，该指标越小表示模型性能越好。
        
        Parameters
        ----------
            net_param:设置模型的参数   dictionary
            train_param:训练是的参数   dictionary

        """
        self.net_param = net_param
        self.train_param = train_param
        self.Version = "V6"
        # 创建模型
        self.net = autoencoder.AutoEncoder(1, 128, (114, 100), 0.3)

        self.losser = torch.nn.MSELoss(reduction='mean')
        # matric【CER】
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # 一个step的训练步骤
        inputs, labels = batch
        features, inputs_ = self.net(inputs)

        loss = self.losser(inputs_, inputs)

        self.log("mseloss", loss)

        return {"loss" : loss}


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        features, inputs_ = self.net(inputs)

        return {"inputs_": inputs_, "inputs": inputs}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        inputs_ = []
        inputs = []
        for output in outputs:
            inputs_ += (output["inputs_"]).cpu().numpy().tolist()
            inputs += (output["inputs"]).cpu().numpy().tolist()
        
        inputs_ = torch.from_numpy(np.array(inputs_)).type(torch.float32).cuda()
        inputs = torch.from_numpy(np.array(inputs)).type(torch.float32).cuda()

        loss = self.losser(inputs_, inputs)
        self.log("result_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
        return optimizer