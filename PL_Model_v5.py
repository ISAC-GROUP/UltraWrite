# ========================================== #
#  this version is a seq2seq framework model #
# ========================================== #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from models import lenet_lstm, Seq2Seq
import torchmetrics
from ctc_decoder import *
from cmath import inf, nan

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
        self.Version = "V5"
        # 创建模型
        self.encoder = Seq2Seq.EncoderLSTM(
            embed_input_nc=self.net_param["embed_input_nc"],
            embed_output_size=self.net_param["embed_output_size"],
            embed_input_shape=self.net_param["embed_input_shape"],
            dropout=self.net_param["dropout"],
            input_size=self.net_param["input_size"],
            Hidden_size=self.net_param["hidden_size"],
            num_layer=self.net_param["num_layer"],
            directions=self.net_param["directions"],
            output_size=self.net_param["encoder_output_size"]
        )

        # 生成损失函数                       
        self.CTCLoss = nn.CTCLoss(blank=self.train_param['BLANK'], zero_infinity=True)

        # matric【CER】
        self.metric = torchmetrics.CharErrorRate()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # 一个step的训练步骤
        inputs, labels, inputs_lengths, labels_lengths = batch

        inputs_lengths = inputs_lengths.cpu()
        encoder_outputs, encoder_hiddens = self.encoder(inputs, inputs_lengths)
        inputs_lengths = inputs_lengths.cuda()

        encoder_outputs = F.log_softmax(encoder_outputs, 2)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)        # 交换第一和第二维 【T, N, C】

        ctcloss = self.CTCLoss(encoder_outputs, labels, inputs_lengths, labels_lengths)

        self.log("ctc_loss", ctcloss)
        if(ctcloss == inf or ctcloss == nan):
            print("nan or inf")

        return {"loss" : ctcloss}

    def int2char(self, label) -> str:
        result = ""
        for l in label:
            if l >= len(self.train_param["CHARS"]):       # 当前值为padding的
                continue
            else:
                result += self.train_param["CHARS"][l]  # 当前值是正常值
        
        return result

    def validation_step(self, batch, batch_idx):
        source_inputs, labels, source_lengths, label_lengths = batch
        source_lengths = source_lengths.cpu()
        # out = self(x, x_lengths)    # [b, ts, cln]
        encoder_outputs, encoder_hiddens = self.encoder(source_inputs, source_lengths)
        
        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.cpu().detach().numpy()
        
        labels = labels.cpu().detach().numpy()    # [b, max_l]  pad的部分为blank，对应的数值为26
        
        encoder_outs = []
        for i in range(len(source_lengths)):
            mat = encoder_outputs[i]
            mat = mat[:source_lengths[i], :]
            encoder_outs.append(best_path(mat, self.train_param["CHARS"]))

        label_outs = []
        for label in labels:
            label = self.int2char(label)
            label_outs.append(label)

        return {"encoder_outs":encoder_outs, "labels":label_outs}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        encoder_outs = []
        labels = []
        for output in outputs:
            encoder_outs += output["encoder_outs"]
            labels += output["labels"]

        with open(os.path.join(os.getcwd(), self.train_param["save_result"]), "a+") as f:
            f.write("==================================")
            f.write("\n")
            for i in range(len(encoder_outs)):
                f.write(encoder_outs[i] + " " + labels[i] + "\n")
            
        f.close()

        self.metric.reset()
        encoder_CER = self.metric(encoder_outs, labels)
        self.metric.reset()
        self.log("encoder_CER", encoder_CER)
    
    def configure_optimizers(self):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.train_param["lr"])
        return encoder_optimizer