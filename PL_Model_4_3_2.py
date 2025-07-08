# ====================================================== #
#  for paper 4.3.2  #
# ====================================================== #
import os
from matplotlib.pyplot import ylabel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from models import Seq2Seq, model432
import torchmetrics
from ctc_decoder import *

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
        self.best_cer = 1.0
        self.best_wer = 1.0
        self.log_path = log_path
        self.Version = "4-3-2"
        # 创建模型
        self.net = model432.EncoderLSTM(
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
            self.net.embed = feature_extractor
            for name, param in self.net.named_parameters():
                if "embed" in name:
                    param.requires_grad = False

        # 生成损失函数                       
        self.ctcloss = nn.CTCLoss(blank=self.train_param['BLANK'], zero_infinity=True)

        # matric【W_acc】
        self.CER = torchmetrics.CharErrorRate()
        self.WER = torchmetrics.WordErrorRate()
        self.save_hyperparameters()

    def forward(self, x, x_lengths):
        y, h = self.net(x, x_lengths)  # 【b, ts, cln】
        return y, h
    
    def training_step(self, batch, batch_idx):
        # 一个step的训练步骤
        x, y, x_lengths, y_lengths = batch

        x_lengths = x_lengths.cpu()

        pre_out, _ = self.net(x, x_lengths)
        
        x_lengths = x_lengths.cuda()
        log_pre_out = F.log_softmax(pre_out, 2)

        log_pre_out = log_pre_out.permute(1, 0, 2)        # 交换第一和第二维 【T, N, C】

        loss = self.ctcloss(log_pre_out, y, x_lengths, y_lengths)

        self.log("ctc_loss", loss)
        return {"loss" : loss}

    def int2char(self, label) -> str:
        result = ""
        for l in label:
            if l >= len(self.train_param["CHARS"]):       # 当前值为padding的
                continue
            else:
                result += self.train_param["CHARS"][l]  # 当前值是正常值
        
        return result

    def validation_step(self, batch, batch_idx):
        """
        一个 step 的验证， 将预测结果和标签以字典方式返回，之后再epoch结束之后，统一
        计算metric

        将每一个batch的数据进行验证，模型输出为尺寸【BatchSize，TimeStep，ClassNum】
        的tensor类型数据，将每一个样本输出【TimeStep， ClassNum】传入decoder进行解码
        出对应的输出，为str。 需要输入CHARS【A~Z】

        需要先将out转化为ndarray格式

        同时需要将y从数字标签转为对应字符串
        """
        x, y, x_lengths, y_lengths = batch
        x_lengths = x_lengths.cpu()
        out, _ = self.net(x, x_lengths)
        
        out = F.log_softmax(out, 2)
        out = out.cpu().detach().numpy()
        y = y.cpu().detach().numpy()    # [b, max_l]  pad的部分为blank，对应的数值为26

        pre_out = []
        for i in range(len(x_lengths)):
            mat = out[i]
            mat = mat[:x_lengths[i], :]
            pre_out.append(best_path(mat, self.train_param["CHARS"]))

        labels = []
        for label in y:
            label = self.int2char(label)
            labels.append(label)

        return {"pre_out":pre_out, "label":labels}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        pres = []
        labels = []
        for output in outputs:
            pres += output["pre_out"]
            labels += output["label"]

        self.CER.reset()
        char_error_rate = self.CER(pres, labels)
        self.WER.reset()
        word_error_rate = self.WER(pres, labels)

        if char_error_rate < self.best_cer:
            self.best_cer = char_error_rate
            self.best_wer = word_error_rate
            with open(os.path.join(os.getcwd(), self.log_path), "w") as f:
                for i in range(len(pres)):
                    f.write(pres[i] + " " + labels[i] +" " + str(pres[i] == labels[i]) + "\n")
                
            f.close()

        self.log("CER", char_error_rate)
        self.log("WER", word_error_rate)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_param["lr"])
        return optimizer