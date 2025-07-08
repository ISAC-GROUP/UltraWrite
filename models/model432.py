from . import Lenet5, lstm, Alexnet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

class EncoderLSTM(nn.Module):
    def __init__(
        self, 
        embed_input_nc = 1,
        embed_output_size = 128,
        embed_input_shape = [114, 100],
        dropout=0.3,
        input_size=128, 
        Hidden_size=512, 
        num_layer=2, 
        directions=True, 
        output_size=27,
    ) -> None:
        super(EncoderLSTM, self).__init__()
        ################
        #### Params ####
        ################
        self.embed_input_nc = embed_input_nc
        self.embed_output_size = embed_output_size
        self.embed_input_shape = embed_input_shape
        self.dropout = dropout
        self.input_size = input_size                            # default = 128
        self.Hidden_size = Hidden_size                          # default = 512
        self.num_layer = num_layer                              # default = 2
        self.directions = directions                            # default = False  

        if self.directions is True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.output_size = output_size
        ###############
        #### Model ####
        ###############
        # backbone确定实验
        self.embed = Lenet5.LeNet5()
 
        # RNN
        # self.rnn = nn.RNN(self.input_size, self.Hidden_size, self.num_layer, batch_first=True)
        # BiRNN
        # self.birnn = nn.RNN(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        # GRU
        # self.gru = nn.GRU(self.input_size, self.Hidden_size, self.num_layer, batch_first=True)
        # BiGRU
        # self.bigru = nn.GRU(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        # LSTM
        # self.lstm = nn.LSTM(self.input_size, self.Hidden_size, self.num_layer, batch_first=True)
        # BiLSTM
        self.bilstm = nn.LSTM(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)

        # self.out = nn.Linear(in_features=self.Hidden_size, out_features=self.output_size)
        self.biout = nn.Linear(in_features=self.Hidden_size*self.num_directions, out_features=self.output_size)


    def forward(self, source_inputs, source_lengths):
        # @_embedding
        bs, ts, cn, h, w = source_inputs.shape
        embed_inputs = source_inputs.contiguous().view(bs*ts, cn, h, w)
        embed_outputs = self.embed(embed_inputs)
        embed_outputs = embed_outputs.contiguous().view(bs, ts, -1) 

        # @_lstm
        # 输出的hidden为非padding部分的最后一个时间步的隐藏层输出
        lstm_inputs = embed_outputs
        lstm_inputs_packed = nn.utils.rnn.pack_padded_sequence(lstm_inputs, lengths=source_lengths, batch_first=True, enforce_sorted=True)
        
        # rnn
        # lstm_outputs_packed, hiddens = self.rnn(lstm_inputs_packed, None)
        # birnn
        # lstm_outputs_packed, hiddens = self.birnn(lstm_inputs_packed, None)
        # gru
        # lstm_outputs_packed, hiddens = self.gru(lstm_inputs_packed, None)
        # bigru
        # lstm_outputs_packed, hiddens = self.bigru(lstm_inputs_packed, None)
        # lstm
        # lstm_outputs_packed, hiddens = self.lstm(lstm_inputs_packed, None)
        # bilstm
        lstm_outputs_packed, hiddens = self.bilstm(lstm_inputs_packed, None)
        

        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs_packed, batch_first=True)
        
        # @_fc
        fc_inputs = lstm_outputs
        bs, ts, _ = fc_inputs.shape
        fc_inputs = fc_inputs.contiguous().view(bs*ts, -1)
        # linear
        # outputs = self.out(fc_inputs)
        # Bilinear
        outputs = self.biout(fc_inputs)
        outputs = outputs.contiguous().view(bs, ts, -1)
        return outputs, hiddens
