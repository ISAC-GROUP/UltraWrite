# ========================================== #
#  this version is a seq2seq framework model #
# ========================================== #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from models import lenet_lstm, Seq2Seq, AttenDecoder
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

        self.decoder = Seq2Seq.DecoderLSTM(
            input_size=self.net_param["input_size"],
            hidden_size=self.net_param["hidden_size"],
            num_layer=self.net_param["num_layer"],
            directional=self.net_param["directions"],
            num_class=self.net_param["decoder_output_size"],
            embedding_dim=self.net_param["input_size"],
            BOS_Token=self.train_param["BOS"],
            EOS_Token=self.train_param["EOS"]
        )
        # self.decoder = AttenDecoder.Decoder(
        #     decoder_input_size=self.net_param['decoder_input_size'],
        #     decoder_hidden_size=512,
        #     decoder_num_layer=self.net_param['decoder_num_layer'],
        #     decoder_direction=True,
        #     decoder_num_embed=self.net_param['decoder_num_embed'],
        #     decoder_embed_dim=self.net_param['decoder_embed_dim'],
        #     dropout=self.net_param['decoder_dropout'],
        #     decoder_output_size=self.net_param['decoder_output_size'],
        #     atten_mode=self.net_param['decoder_atten_mode'],
        #     pad=self.net_param['decoder_pad'],
        #     BOS_Token=self.net_param['BOS_Token'],
        #     EOS_Token=self.net_param['EOS_Token']
        # )

        # 生成损失函数                       
        self.CTCLoss = nn.CTCLoss(blank=self.train_param['BLANK'])
        self.CELoss = nn.CrossEntropyLoss(ignore_index=self.train_param["PAD"])

        # matric【CER】
        self.metric = torchmetrics.CharErrorRate()
        self.save_hyperparameters()

    def create_targets(self, labels, label_lengths):
        batch_size, timesteps = labels.shape
        device = labels.device

        target_inputs = torch.full(size=(batch_size, timesteps+1), fill_value=self.train_param["PAD"]).to(device)
        target_labels = torch.full(size=(batch_size, timesteps+1), fill_value=self.train_param["PAD"]).to(device)
        target_lengths = (label_lengths.add(1)).to(device)

        target_inputs[:,1:] = labels
        target_inputs[:,0] = self.train_param["BOS"]

        for i in range(batch_size):
            l = label_lengths[i].item()
            target_labels[i, :l] = labels[i,:l]
            target_labels[i, l] = self.train_param["EOS"]

        return target_inputs, target_labels, target_lengths

    def training_step(self, batch, batch_idx):
        # 一个step的训练步骤
        inputs, labels, inputs_lengths, labels_lengths = batch

        inputs_lengths = inputs_lengths.cpu()
        encoder_outputs, encoder_hiddens = self.encoder(inputs, inputs_lengths)
        inputs_lengths = inputs_lengths.cuda()

        target_inputs, target_labels, target_lengths = self.create_targets(labels, labels_lengths)

        decoder_outputs, decoder_lengths = self.decoder(
            target_inputs=target_inputs,
            target_lengths=target_lengths,
            hiddens=encoder_hiddens,
            max_length=None,
            pad=None
        )

        encoder_outputs = F.log_softmax(encoder_outputs, 2)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)        # 交换第一和第二维 【T, N, C】

        b, s, l = decoder_outputs.shape
        decoder_outputs = decoder_outputs.contiguous().view(b*s, -1)
        target_labels = target_labels.contiguous().view(-1)

        ctcloss = self.CTCLoss(encoder_outputs, labels, inputs_lengths, labels_lengths)
        celoss = self.CELoss(decoder_outputs, target_labels)

        loss = ((1 - self.train_param['lambda']) * celoss) + (self.train_param['lambda'] * ctcloss)

        self.log("ctc_loss", ctcloss)
        self.log("ce_loss", celoss)
        self.log('loss', loss)
        return {"loss":loss}

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
        
        decoder_outputs, decoder_lengths = self.decoder(
            target_inputs=None,
            target_lengths=None,
            hiddens=encoder_hiddens,
            max_length=self.train_param["max_length"],
            pad=self.train_param["PAD"]
        )
        # decoder_outputs, decoder_lengths = self.decoder(
        #     trg_inputs=None,
        #     trg_len=None,
        #     source_len=source_lengths,
        #     encoder_outputs=encoder_outputs,
        #     encoder_last_hidden=encoder_hiddens[0],
        #     max_length=self.train_param['max_length']
        # )

        encoder_outputs = F.log_softmax(encoder_outputs, 2)
        encoder_outputs = encoder_outputs.cpu().detach().numpy()

        decoder_outputs = decoder_outputs.cpu().detach().numpy()
        
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
        
        decoder_outs = []
        for word in decoder_outputs:
            word = self.int2char(word)
            decoder_outs.append(word)

        return {"encoder_outs":encoder_outs, "decoder_outs":decoder_outs, "labels":label_outs}
    
    def validation_epoch_end(self, outputs):
        """
        一个epoch的验证结束，将所有的预测结果和对应的标签计算metric
        """
        encoder_outs = []
        decoder_outs = []
        labels = []
        for output in outputs:
            encoder_outs += output["encoder_outs"]
            decoder_outs += output["decoder_outs"]
            labels += output["labels"]

        with open(os.path.join(os.getcwd(), self.train_param["save_result"]), "a+") as f:
            f.write("==================================")
            f.write("\n")
            for i in range(len(encoder_outs)):
                f.write(encoder_outs[i] + " " + decoder_outs[i] + " " + labels[i] + "\n")
            
        f.close()

        self.metric.reset()
        encoder_CER = self.metric(encoder_outs, labels)
        self.metric.reset()
        decoder_CER = self.metric(decoder_outs, labels)
        self.log("encoder_CER", encoder_CER)
        self.log("decoder_CER", decoder_CER)
    
    def configure_optimizers(self):
        # encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.train_param["lr"])
        # decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.train_param["lr"])
        # return [encoder_optimizer, decoder_optimizer], []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_param['lr'])
        return optimizer