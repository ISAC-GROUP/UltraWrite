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
    def __init__(self, net_param, train_param, feature_extractor=None, log_path=None):
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
        self.Version = "double_check"
        self.best_cer = 1.0
        self.best_wer = 1.0
        self.log_path = log_path
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
        if feature_extractor is not None:
            self.encoder.embed = feature_extractor
            for name, param in self.encoder.named_parameters():
                if "embed" in name:
                    param.requires_grad = False


        # 生成损失函数                       
        self.CTCLoss = nn.CTCLoss(blank=self.train_param['BLANK'], zero_infinity=True)

        # matric【CER】
        self.CER = torchmetrics.CharErrorRate()
        self.WER = torchmetrics.WordErrorRate()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # 一个step的训练步骤
        inputs, labels, inputs_lengths, labels_lengths, ids = batch

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
        source_inputs, labels, source_lengths, label_lengths, ids = batch
        source_lengths = source_lengths.cpu()
        # out = self(x, x_lengths)    # [b, ts, cln]
        encoder_outputs, encoder_hiddens = self.encoder(source_inputs, source_lengths)
        

        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.cpu().detach().numpy()
        
        labels = labels.cpu().detach().numpy()    # [b, max_l]  pad的部分为blank，对应的数值为26
        

        encoder_outs = []
        # for mat in encoder_outputs:
        #     encoder_outs.append(best_path(mat, self.train_param["CHARS"]))
        for i in range(len(source_lengths)):
            mat = encoder_outputs[i]
            mat = mat[:source_lengths[i], :]
            encoder_outs.append(best_path(mat, self.train_param["CHARS"]))

        label_outs = []
        for label in labels:
            label = self.int2char(label)
            label_outs.append(label)
        
        ids = ids.tolist()

        return {"encoder_outs":encoder_outs,"labels":label_outs, "ids":ids}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        encoder_outs = []
        labels = []
        ids = []
        for output in outputs:
            encoder_outs += output["encoder_outs"]
            labels += output["labels"]
            ids += output["ids"]

        self.CER.reset()
        char_error_rate = self.CER(encoder_outs, labels)
        
        self.WER.reset()
        word_error_rate = self.WER(encoder_outs, labels)
        
        if self.best_cer > char_error_rate:
            self.best_cer = char_error_rate
            self.best_wer = word_error_rate
            with open(os.path.join(os.getcwd(), self.log_path), "w") as f:
                for i in range(len(encoder_outs)):
                    f.write(encoder_outs[i] + " " + labels[i] + " " + str(ids[i]) + " " + str(encoder_outs[i] == labels[i]) + "\n")
                
            f.close()

        self.log("CER", char_error_rate)
        self.log("WER", word_error_rate)
    
    def configure_optimizers(self):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.train_param["lr"])
        return encoder_optimizer