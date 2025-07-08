import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self, 
        embed_input = 26,
        dropout=0.3,
        input_size=32, 
        Hidden_size=64, 
        num_layer=2, 
        directions=True
    ) -> None:
        super(Encoder, self).__init__()
        ################
        #### Params ####
        ################
        self.embed_input = embed_input
        self.dropout = dropout
        self.input_size = input_size                            # default = 32
        self.Hidden_size = Hidden_size                          # default = 64
        self.num_layer = num_layer                              # default = 2
        self.directions = directions                            # default = False  

        if self.directions is True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        ###############
        #### Model ####
        ###############
        self.embed = nn.Embedding(self.embed_input, self.input_size)
        
        self.lstm = nn.LSTM(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)


    def forward(self, source_inputs, source_lengths):
        # @_embedding
        # source_inputs = batch_size, T
        embed_outputs = self.embed(source_inputs)       # batch_size, T, N

        # @_lstm
        # 输出的hidden为非padding部分的最后一个时间步的隐藏层输出
        lstm_inputs = embed_outputs
        lstm_inputs_packed = nn.utils.rnn.pack_padded_sequence(lstm_inputs, lengths=source_lengths, batch_first=True, enforce_sorted=False)
        lstm_outputs_packed, hiddens = self.lstm(lstm_inputs_packed, None)
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs_packed, batch_first=True)
        
        return lstm_outputs, hiddens


net = Encoder(27, 0.3, 32, 64, 2, True)
print(net)
inputs = [torch.tensor([1,2,3,4]), torch.tensor([5,6,7,8,9])]
inputs_lengths = torch.tensor([4, 5])
inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=26)
outputs,_ = net(inputs, inputs_lengths)
print(outputs.shape)
exit(0)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(input_size=hidden_size + embed_size, hidden_size=hidden_size,
                          num_layers=n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

class Seq2Seq(nn.Module):
    def __init__(
        self, 
        encoder_input_size, 
        encoder_embed_size, 
        encoder_hidden_size, 
        encoder_nlayer,
        encoder_dropout,
        decoder_embed_size,
        decoder_hidden_size,
        decoder_output_size,
        decoder_nlayer,
        decoder_dropout
    ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_input_size, encoder_embed_size, encoder_hidden_size, encoder_nlayer, encoder_dropout)
        self.decoder = Decoder(decoder_embed_size, decoder_hidden_size, decoder_output_size, decoder_nlayer, decoder_dropout)
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = batch, T, inputsize
        # trg = batch, T, outputsize
        batch_size = src.size(0)
        max_len = src.size(1)
        vocab_size = self.decoder.output_size

        encoder_outputs, hiddens = self.encoder(src, None)      # encoder_outputs -> B T D*H          hiddens -> D*num_layer N H
        hiddens = hiddens[:self.decoder.n_layers]               # num_layer N H

        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        output = Variable(trg.data[:, 0, :])  # sos

        for t in range(1, max_len):
            output, hiddens, attn_weights = self.decoder(
                    output, hiddens, encoder_outputs)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs