from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# atten_net = Attention(mode='general', hidden_size=128)
# encoder_outputs = torch.randn(size=(1, 16, 128))
# decoder_output = torch.randn(size=(1, 1, 128))
# atten_weight = atten_net(encoder_outputs, decoder_output)
# atten_weight = atten_weight.unsqueeze(1)
# context = torch.bmm(atten_weight, encoder_outputs)
# print(context.shape)
# exit(0)

class Decoder(nn.Module):
    def __init__(
        self,
        decoder_input_size,
        decoder_hidden_size,
        decoder_num_layer,
        decoder_direction,
        decoder_num_embed,
        decoder_embed_dim,
        dropout,
        decoder_output_size,
        atten_mode,
        pad,
        BOS_Token,
        EOS_Token,
    ) -> None:
        super(Decoder, self).__init__()
        '''
        notes
        -----
            num_embed = output_size + 1
            num_embed include [EOS] and [BOS]
            output_size just include [EOS]
        '''
        self.decoder_input_size = decoder_input_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layer = decoder_num_layer
        self.decoder_direction = decoder_direction
        self.decoder_num_embed = decoder_num_embed 
        self.decoder_embed_dim = decoder_embed_dim          # embed_dim == rnn_input_size
        self.dropout = dropout
        self.decoder_output_size = decoder_output_size    
        self.atten_mode = atten_mode
        self.pad = pad  
        self.BOS_Token = BOS_Token
        self.EOS_Token = EOS_Token

        self.embed = nn.Embedding(self.decoder_num_embed, self.decoder_embed_dim)

        self.rnn = nn.GRU(
            input_size= self.decoder_input_size,
            hidden_size= self.decoder_hidden_size,
            num_layers=self.decoder_num_layer,
            batch_first=True,
            dropout=self.dropout,
            bidirectional = self.decoder_direction 
        )

        self.attention_net = Attention(mode=self.atten_mode, hidden_size=self.decoder_hidden_size)

        self.fc = nn.Linear(self.decoder_hidden_size*2, self.decoder_output_size)

    def train_forward(
        self, 
        trg_inputs:torch.Tensor, 
        trg_len:torch.Tensor, 
        source_len:torch.Tensor,
        encoder_outputs:torch.Tensor, 
        encoder_last_hiddens:torch.Tensor
    ) -> torch.Tensor:
        '''
        Parameters
        ----------
            trg_inputs(tensor): batchSize x timeStep
            trg_len(tensor) : batchSize x 1
            encoder_outputs(tensor): batchSize x timestep x encoder_output_shape
            encoder_last_hiddens(tensor): (num_layer) x batchSize x hidden_size 
        '''
        # 数据准备
        device = trg_inputs.device
        batch_size, time_step = trg_inputs.shape
        outputs = torch.full(size=(batch_size, time_step, self.decoder_output_size), fill_value=self.pad, dtype=torch.float32).to(device)

        # 以单个样本为单位进行计算
        for batch_idx in range(batch_size):
            times = trg_len[batch_idx].item()        # label len of each sample
            
            h = encoder_last_hiddens[:,batch_idx,:]                   
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])  # [num_layers x 1 x hidden_size]
            
            encoder_output = encoder_outputs[batch_idx].contiguous().view(1, encoder_outputs.shape[1], encoder_outputs.shape[2])      
            encoder_output = encoder_output[:,:source_len[batch_idx],:]
            # encoder_output shape = [1 x timestep x encoder_out_shape(128)] 
            for t in range(times):
                input = trg_inputs[batch_idx][t]
                embed_output = self.embed(input).contiguous().view(1, 1, -1)
                
                decoder_output, h = self.rnn(embed_output, h)            
                # decoder_output = decoder_output[:,:,:self.rnn_hidden_size] + decoder_output[:,:,self.rnn_hidden_size:]  
                # decoder_output shape = [1, 1, hidden_size]

                atten_weight = self.attention_net(encoder_output, decoder_output)   # [1 x timeStep]   
                atten_weight = atten_weight.unsqueeze(1)                            # [1, 1, timeStep]
                context = torch.bmm(atten_weight, encoder_output)                   # [1, 1, hidden_size]
                fc_input = torch.cat((decoder_output, context), dim=2)              # [1, 1, hidden_size * 2]
                fc_output = self.fc(fc_input)
                
                outputs[batch_idx][t] = fc_output
        
        return outputs
    
    def test_forward(
        self,
        source_len:torch.Tensor,
        encoder_outputs:torch.Tensor,
        encoder_last_hidden:torch.Tensor,
        max_length:int
    ) -> torch.Tensor:
        '''
        Parameters
        ----------
            encoder_outputs(tensor): batchSize x timestep x encoder_output_shape
            encoder_last_hiddens(tensor): (num_layer) x batchSize x hidden_size 
            max_length(int): max length of the output
        '''


        device = encoder_outputs.device
        batch_size = encoder_outputs.shape[0]

        outputs = torch.full(size=(batch_size, max_length), fill_value=self.pad).to(device)

        for batch_idx in range(batch_size):
            # 每次计算一个样本
            input = torch.tensor([self.BOS_Token])              # first time step input

            h = encoder_last_hidden[:,batch_idx,:]                       # first time step hidden input
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])

            encoder_output = encoder_outputs[batch_idx].contiguous().view(1, encoder_outputs.shape[1], encoder_outputs.shape[2])
            encoder_output = encoder_output[:,:source_len[batch_idx],:]
            for t in range(max_length):
                input = input.to(device)
                h = h.to(device)

                embed_output = self.embed(input).contiguous().view(1, 1, -1)        # [1 x 1 x inputSize]

                decoder_output, h = self.rnn(embed_output, h)                       # [1 x 1 x hidden_size]

                atten_weight = self.attention_net(encoder_output, decoder_output)   # [1 x timeStep]
                atten_weight = atten_weight.unsqueeze(1)                            # [1 x 1 x timeStep]

                context = torch.bmm(atten_weight, encoder_output)                   # [1 x 1 x hiddenSize]

                fc_input = torch.cat((decoder_output, context), dim=2)              # [1 x 1 x 2*hiddenSize]

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
        source_len:torch.Tensor,
        encoder_outputs:torch.Tensor, 
        encoder_last_hidden:torch.Tensor,
        max_length:int
    ) -> torch.Tensor:

        if self.training is True:
            return self.train_forward(trg_inputs, trg_len, source_len, encoder_outputs, encoder_last_hidden)

        elif self.training is False:
            return self.test_forward(source_len, encoder_outputs, encoder_last_hidden, max_length)


# model = Decoder(
#     decoder_input_size=128, 
#     decoder_hidden_size=512, 
#     decoder_num_layer=2, 
#     decoder_direction=False, 
#     decoder_num_embed=28,
#     decoder_embed_dim=128, 
#     dropout=0.3, 
#     decoder_output_size=27, 
#     atten_mode='general', 
#     pad=27,
#     BOS_Token=27,
#     EOS_Token=26)

# model.eval()
# # print(model.training)
# # exit(0)
# trg_inputs = torch.randint(0, 27, size=(3, 10))
# trg_len = torch.tensor([10, 7, 5])
# encoder_outputs = torch.randn(size=(3, 12, 512))
# encoder_last_hidden = torch.randn(size=(2, 3, 512))

# outputs = model(trg_inputs, trg_len, encoder_outputs, encoder_last_hidden, 10)
# print(outputs.shape)