import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self, 
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layer,
        rnn_direction,
        num_embed,
        embed_dim, 
        dropout=0.5
    ) -> None:
        super(Encoder, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layer = rnn_num_layer
        self.rnn_direction = rnn_direction
        self.num_embed = num_embed 
        self.embed_dim = embed_dim          # embed_dim == rnn_input_size
        self.dropout = dropout

        self.embed = nn.Embedding(num_embeddings=self.num_embed, embedding_dim=self.embed_dim)
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size, 
            num_layers=self.rnn_num_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.rnn_direction
        )

    def forward(self, src, src_len, hidden=None):
        '''
        Notes
        -----
            hidden只有最后一个时间步。如若要使用attention的话
            则需要所有时间步的output。并且只需要将direction进行合并
        '''
        # src [batch_size, timesteps]
        # src_len [batch_size]
        embedded = self.embed(src)      # [batch_size, timesteps, embed_dim]
        lstm_input = embedded
        lstm_input = nn.utils.rnn.pack_padded_sequence(lstm_input, src_len, batch_first=True, enforce_sorted=False)
        lstm_output, hidden = self.lstm(lstm_input, hidden)  
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True, padding_value=0.0)
        # lstm_output [batch_size, timesteps, direction_num * hidden_size]
        # hidden [num_layer * direction_num, batch_size, direction_num * hidden_size]
        # sum bidirectional outputs
        outputs = lstm_output[:, :, :self.rnn_hidden_size] + lstm_output[:, :, self.rnn_hidden_size:]
        return outputs , hidden

class Attention(nn.Module):
    r'''
    Notes
    -----
        该模块作为模型的其中一个部分插入到decoder中。在前向传播时，传入的
        参数分别为：
        1.encoder_outputs: 为encoder的每一个时间步的输出，尺寸为：
        [batch_size, timesteps, hidden_size]，encoder的输出最后一个维度
        若是双向LSTM，则需要将前后合并。
        2.decoder_output: 为decoder当前时间步的输出，尺寸为：[batch_size, 1,
        hidden_size]。同理如若为双向，则需要将前后合并
    
    '''
    def __init__(self, mode, hidden_size) -> None:
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode

        if self.mode not in ["dot", "general", "concat"]:
            raise ValueError(self.mode, "is not a appropriate attention mode.")

        if self.mode == "general":
            self.atten = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.mode == "concat":
            self.atten = nn.Linear(self.hidden_size*2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(self.hidden_size))  

    def forward(self, encoder_outputs, decoder_output):
        if self.mode == "general":
            score = self.general_score(encoder_outputs, decoder_output)
        
        if self.mode == "concat":
            score = self.concat_score(encoder_outputs, decoder_output)
        
        if self.mode == "dot":
            score = self.dot_score(encoder_outputs, decoder_output)
        
        attention_vector = F.softmax(score, dim = -1)
        return attention_vector
        
    def dot_score(self, encoder_outputs, decoder_output):
        score = torch.sum(encoder_outputs * decoder_output, dim = -1)
        return score
    
    def general_score(self, encoder_outputs, decoder_output):
        atten_general = self.atten(encoder_outputs)
        score = torch.sum(decoder_output * atten_general, dim=-1)
        return score

    def concat_score(self, encoder_outputs, decoder_output):
        temp_decoder_outputs = decoder_output.expand(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        atten_concat = self.atten(
            torch.cat((temp_decoder_outputs, encoder_outputs), dim=2)
        )        

        score = torch.sum(self.v * atten_concat, dim=2)
        return score

# atten_net = Attention(mode='general', hidden_size=16)
# encoder_outputs = torch.randn(size=(1, 4, 16))
# decoder_output = torch.randn(size=(1, 1, 16))
# atten_weight = atten_net(encoder_outputs, decoder_output)
# print(atten_weight.shape)
# exit(0)

class Decoder(nn.Module):
    def __init__(
        self,
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layer,
        rnn_direction,
        num_embed,
        embed_dim,
        dropout,
        output_size,
        mode,
        pad
    ) -> None:
        super(Decoder, self).__init__()
        '''
        notes
        -----
            num_embed = output_size + 1
            num_embed include [EOS] and [BOS]
            output_size just include [EOS]
        '''
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layer = rnn_num_layer
        self.rnn_direction = rnn_direction
        self.num_embed = num_embed 
        self.embed_dim = embed_dim          # embed_dim == rnn_input_size
        self.dropout = dropout
        self.output_size = output_size    
        self.mode = mode
        self.pad = pad  

        self.embed = nn.Embedding(self.num_embed, self.embed_dim)

        self.lstm = nn.LSTM(
            input_size = self.rnn_input_size,
            hidden_size = self.rnn_hidden_size,
            num_layers = self.rnn_num_layer,
            batch_first = True,
            dropout=self.dropout,
            bidirectionals = self.rnn_direction
        )

        self.attention_net = Attention(mode=self.mode, hidden_size=self.rnn_hidden_size)

        self.fc = nn.Linear(self.rnn_hidden_size*2, self.output_size)

    def train_forward(
        self, 
        trg_inputs:torch.Tensor, 
        trg_len:torch.Tensor, 
        encoder_outputs:torch.Tensor, 
        encoder_last_hiddens:torch.Tensor
    ) -> torch.Tensor:

        # 数据准备
        device = trg_inputs.device
        batch_size, time_step = trg_inputs.shape
        outputs = torch.full(size=(batch_size, time_step, self.output_size), fill_value=self.pad).to(device)

        # 以单个样本为单位进行计算
        for batch_idx in range(batch_size):
            times = trg_len[batch_idx].item()        # label len of each sample
            
            h = encoder_last_hiddens[0][:,batch_idx,:]                   
            c = encoder_last_hiddens[1][:,batch_idx,:]
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])
            c = c.contiguous().view(c.shape[0], 1, c.shape[1])
            
            encoder_output = encoder_outputs[batch_idx,:times].contiguous().view(1, times, encoder_outputs.shape[2])
            # encoder_output shape = [1, timestep, hidden_size] 
            for t in range(times):
                input = trg_inputs[batch_idx][t]
                embed_output = self.embed(input).contiguous().view(1, 1, -1)
                
                decoder_output, (h, c) = self.lstm(embed_output, (h, c))            
                decoder_output = decoder_output[:,:,:self.rnn_hidden_size] + decoder_output[:,:,self.rnn_hidden_size:]  
                # decoder_output shape = [1, 1, hidden_size]

                atten_weight = self.attention_net(encoder_output, decoder_output)   
                atten_weight = atten_weight.unsqueeze(1)
                # atten_weight shape = [1, 1, timestep]
                context = torch.bmm(atten_weight, encoder_output)
                # context shape = [1, 1, hidden_size]
                fc_input = torch.cat((decoder_output, context), dim=2)      # [1, 1, hidden_size * 2]
                fc_output = self.fc(fc_input)
                
                outputs[batch_idx][t] = fc_output
        
        return outputs
    
    def test_forward(
        self,
        encoder_outputs:torch.Tensor,
        encoder_last_hidden:torch.Tensor,
        trg_len:torch.Tensor,
        max_length
    ) -> torch.Tensor:
        device = encoder_outputs.device
        batch_size = encoder_outputs.shape[0]

        outputs = torch.full(size=(batch_size, max_length), fill_value=self.pad).to(device)

        for batch_idx in range(batch_size):
            # 每次计算一个样本
            input = torch.tensor([self.BOS_Token])              # first time step input

            h = encoder_last_hidden[0][:,batch_idx,:]                       # first time step hidden input
            c = encoder_last_hidden[1][:,batch_idx,:]                       # first time step cell station
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])
            c = c.contiguous().view(c.shape[0], 1, c.shape[1])

            encoder_output = encoder_outputs[batch_idx,:trg_len[batch_idx]].contiguous().view(1, trg_len[batch_idx], encoder_outputs.shape[2])

            for t in range(max_length):
                input = input.to(device)
                h = h.to(device)
                c = c.to(device)

                embed_output = self.embed(input).contiguous().view(1, 1, -1)

                decoder_output, (h, c) = self.lstm(embed_output, (h, c))       # decoder_output shape = [1, 1, hidden_size*2]
                decoder_output = decoder_output[:,:,:self.rnn_hidden_size] + decoder_output[:,:,self.rnn_hidden_size:]

                atten_weight = self.attention_net(encoder_output, decoder_output)
                atten_weight = atten_weight.unsqueeze(1)

                context = torch.bmm(atten_weight, encoder_output)

                fc_input = torch.cat((decoder_output, context), dim=2)

                fc_output = self.fc(fc_input)

                _, topi = fc_output.data.topk(1)
                outputs[batch_idx][t] = topi.item()
                
                if topi.item() == self.EOS_Token:
                    break
                
                input = torch.tensor([topi.item()])

        return outputs

    def forward(
        self, 
        trg_inputs:torch.Tensor, 
        trg_len:torch.Tensor, 
        encoder_outputs:torch.Tensor, 
        encoder_last_hidden:torch.Tensor,
        max_length:int
    ) -> torch.Tensor:
        if self.train is True:
            return self.train_forward(trg_inputs, trg_len, encoder_outputs, encoder_last_hidden)

        elif self.train is False:
            return self.test_forward(encoder_outputs, encoder_last_hidden, trg_len, max_length)
        

class SC_model(nn.Module):
    def __init__(
        self,
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layer,
        rnn_direction,
        num_class,
        dropout,
        atten_mode,
        pad,
        max_length
    ):
        super().__init__()
        
        self.max_length = max_length

        self.encoder = Encoder(
            rnn_input_size=rnn_input_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layer=rnn_num_layer,
            rnn_direction=rnn_direction,
            num_embed=num_class+1,                # include [EOS]
            embed_dim=rnn_input_size,
            dropout=dropout
        )

        self.decoder = Decoder(
            rnn_input_size=rnn_input_size,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layer=rnn_num_layer,
            rnn_direction=rnn_direction,
            num_embed=num_class+2,                # include [BOS] and [EOS]
            embed_dim=rnn_input_size,
            dropout=dropout,
            output_size=num_class+1,              # include [EOS]
            mode=atten_mode,
            pad=pad
        )


    def forward(self, src_inputs, src_len, trg_inputs, trg_len):
        encoder_outputs, encoder_last_hidden = self.encoder(src_inputs, src_len)
        decoder_outputs = self.decoder(trg_inputs, trg_len, encoder_outputs, encoder_last_hidden, self.max_length)

        return decoder_outputs