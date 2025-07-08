# ========================================== #
#  for paper 4.3.4 #
# ========================================== #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from models import model434dann
import torchmetrics
from ctc_decoder import *
from cmath import inf, nan
from torch.autograd import Variable

class pl_model(pl.LightningModule):
    def __init__(self, net_param, train_param, log_path=None):
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
        self.Version = "4-3-4 dann"
        # 创建模型
        self.net = model434dann.DannModel(
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

        self.epoch_iteration = 106

        # 生成损失函数                       
        self.CTCLoss = nn.CTCLoss(blank=self.train_param['BLANK'], zero_infinity=True)              # ctc 损失
        self.DomainLoss = nn.NLLLoss()                                                              # dann 域损失

        # matric【CER】
        self.CER = torchmetrics.CharErrorRate()
        self.WER = torchmetrics.WordErrorRate()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # 获取源域与目标域样本
        data_src = batch["data_src"]
        label_src = batch["label_src"]
        data_length_src = batch["data_length_src"]
        label_length_src = batch["label_length_src"]

        data_trg = batch["data_trg"]
        label_trg = batch["label_trg"]
        data_length_trg = batch["data_length_trg"]
        label_length_trg = batch["label_length_trg"]

        # 一个step的训练步骤
        alpha = float(batch_idx / self.epoch_iteration + self.current_epoch)
        # src
        data_length_src = data_length_src.cpu()
        encoder_outputs, encoder_hiddens, domain_output_src = self.net(data_src, data_length_src, alpha=alpha)
        data_length_src = data_length_src.cuda()

        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)        # 交换第一和第二维 【T, N, C】

        # trg
        data_length_trg = data_length_trg.cpu()
        _, _, domain_output_trg = self.net(data_trg, data_length_trg, alpha=alpha)
        # data_length_trg = data_length_trg.cuda()

        bs, ts, _, _, _ = data_src.shape
        bt, tt, _, _, _ = data_trg.shape

        domain_s = torch.zeros(bs*ts, dtype=torch.long).cuda()
        domain_t = torch.ones(bt*tt, dtype=torch.long).cuda()

        dannloss_s = self.DomainLoss(domain_output_src, domain_s)
        dannloss_t = self.DomainLoss(domain_output_trg, domain_t)


        ctcloss = self.CTCLoss(encoder_outputs, label_src , data_length_src, label_length_src)
        
        totalloss = ctcloss + dannloss_s +dannloss_t

        self.log("total_loss", totalloss)
        self.log("ctc_loss", ctcloss)
        self.log("dann_loss_src", dannloss_s)
        self.log("dann_loss_trg", dannloss_t)

        return {"loss" : totalloss}


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
        outputs, hiddens, domain_outputs = self.net(source_inputs, source_lengths, 0)
        
        outputs = F.log_softmax(outputs, 2)
        outputs = outputs.cpu().detach().numpy()
        
        labels = labels.cpu().detach().numpy()    # [b, max_l]  pad的部分为blank，对应的数值为26
        

        outs = []
        # for mat in encoder_outputs:
        #     encoder_outs.append(best_path(mat, self.train_param["CHARS"]))
        for i in range(len(source_lengths)):
            mat = outputs[i]
            mat = mat[:source_lengths[i], :]
            outs.append(best_path(mat, self.train_param["CHARS"]))
            # encoder_outs.append(beam_search(mat, self.chars, beam_width=25, lm=self.lm))

        label_outs = []
        for label in labels:
            label = self.int2char(label)
            label_outs.append(label)

        return {"outs":outs,"labels":label_outs}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        encoder_outs = []
        labels = []
        for output in outputs:
            encoder_outs += output["outs"]
            labels += output["labels"]

        self.CER.reset()
        char_error_rate = self.CER(encoder_outs, labels)
        self.WER.reset()
        word_error_rate = self.WER(encoder_outs, labels)

        if char_error_rate < self.best_cer:
            self.best_cer = char_error_rate
            self.best_wer = word_error_rate
            with open(os.path.join(os.getcwd(), self.log_path), "w") as f:
                for i in range(len(encoder_outs)):
                    f.write(encoder_outs[i] + " " + labels[i] + " " + str(encoder_outs[i] == labels[i]) + "\n")
                
            f.close()

        self.log("CER", char_error_rate)
        self.log("WER", word_error_rate)
    
    def configure_optimizers(self):
        encoder_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.train_param["lr"])
        return encoder_optimizer