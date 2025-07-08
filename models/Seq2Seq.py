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

        
        # self.bigru = nn.GRU(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        self.bigru = nn.LSTM(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        self.out = nn.Linear(in_features=self.Hidden_size*self.num_directions, out_features=self.output_size)

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
        lstm_outputs_packed, hiddens = self.bigru(lstm_inputs_packed, None)
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs_packed, batch_first=True)
        
        # @_fc
        fc_inputs = lstm_outputs
        bs, ts, _ = fc_inputs.shape
        fc_inputs = fc_inputs.contiguous().view(bs*ts, -1)
        outputs = self.out(fc_inputs)
        outputs = outputs.contiguous().view(bs, ts, -1)
        return outputs, hiddens


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        input_size = 128,
        hidden_size = 512,
        num_layer = 2,
        directional = True,
        num_class = 27, # include <EOS>
        embedding_dim = 128,
        BOS_Token = 27,
        EOS_Token = 26
    ) -> None:
        super(DecoderLSTM, self).__init__()
        # save params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.directional = directional
        if self.directional is True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.num_embedding = num_class + 1          # <BOS> is the last dim
        self.embedding_dim = embedding_dim

        self.BOS_Token = BOS_Token
        self.EOS_Token = EOS_Token
        
        self.output_size = num_class                # output without <BOS>

        self.embed = nn.Embedding(num_embeddings=self.num_embedding, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer, batch_first=True, bidirectional=self.directional)
        self.out = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=self.output_size)
    

    def train_forward(
        self,
        target_inputs:torch.Tensor,
        target_lengths:torch.Tensor,
        hiddens:torch.Tensor
    ):
        device = target_inputs.device

        batch_size, timesteps = target_inputs.shape         # [b, t]
        outputs = torch.zeros(size=(batch_size, timesteps, self.output_size), dtype=torch.float)    
        outputs = outputs.to(device)
        
        for batch_idx in range(batch_size):
            times = target_lengths[batch_idx].item()        # label len of each sample
            h = hiddens[0][:,batch_idx,:]                   
            c = hiddens[1][:,batch_idx,:]
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])
            c = c.contiguous().view(c.shape[0], 1, c.shape[1])

            for t in range(times):
                input = target_inputs[batch_idx][t]

                embed_output = self.embed(input).contiguous().view(1, 1, -1)
                lstm_output, (h, c) = self.lstm(embed_output, (h, c))
                fc_output = self.out(lstm_output)
                
                outputs[batch_idx][t] = fc_output
        
        return outputs, target_lengths

    def test_forward(
        self, 
        hiddens:torch.Tensor, 
        max_length:int,
        pad:int,
    ):
        # 注：在测试阶段 target_input 和 target_lengths无需用到
        device = hiddens[0].device
        batch_size = hiddens[0].shape[1]

        outputs = torch.full(size=(batch_size, max_length), fill_value=pad).to(device)
        lengths = torch.zeros(size=(batch_size,), dtype=torch.long).to(device)

        for batch_idx in range(batch_size):
            # 每次计算一个样本
            input = torch.tensor([self.BOS_Token])              # first time step input

            h = hiddens[0][:,batch_idx,:]                       # first time step hidden input
            c = hiddens[1][:,batch_idx,:]                       # first time step cell station
            h = h.contiguous().view(h.shape[0], 1, h.shape[1])
            c = c.contiguous().view(c.shape[0], 1, c.shape[1])

            length = 0
            for t in range(max_length):
                input = input.to(device)
                h = h.to(device)
                c = c.to(device)
                embed_output = self.embed(input).contiguous().view(1, 1, -1)
                lstm_output, (h, c) = self.lstm(embed_output, (h, c))
                fc_output = self.out(lstm_output)
                _, topi = fc_output.data.topk(1)
                outputs[batch_idx][t] = topi.item()
                if topi.item() == self.EOS_Token:
                    break
                length += 1
                input = torch.tensor([topi.item()])
            lengths[batch_idx] = length

        return outputs, lengths

    def forward(self, target_inputs, target_lengths, hiddens, max_length, pad):
        '''
        Notes
        -----
            train_flag is the sign to show model station,
            if is True mean training station, else is testing station.

        Parameters
        ----------
            target_inputs (tensor) :decoder inputs, using in training 
                station. each target's first timestep is <BOS>.
                shape is [batch_size, timesteps]
            
            target_lengths (tensor) :express each targets length. using
                in packed function. just using in training station.

            hiddens (tuple) :hiddens outputs of encoder. the first element
                is hidden state, the second element is cell state.
                each shape is [num_layer*num_directions, batch_size, hidden_size]

            max_lengths (int):limits the maximum length of a word when model
                is testing.
            
            pad (int):pad value

        Returns
        -------
            outputs (tensor):shape is batch_size * max_lengths * output_size in training state;
                shape is batch_size * max_lengths in testing state
        Exemples
        --------
            >>> target_inputs = torch.randn(size=(16, 10))
            >>> target_lengths = torch.full(size=(1,16), 10)
            >>> hiddens = (torch.randn(size=(4, 16, 512)), torch.randn(size=(4, 16, 512))
            >>> net = DecoderLSTM()
            >>> outputs = net(train_inputs, traget_lengths, hiddens, None)
        '''

        if self.training:
            return self.train_forward(
                target_inputs=target_inputs, 
                target_lengths=target_lengths,
                hiddens=hiddens
            )

        else:
            return self.test_forward(
                hiddens=hiddens, 
                max_length=max_length,
                pad=pad
            )

   # def beam_search(self, x, beam_width, eos):
    #     def _inflate(tensor, times, dim):
    #         repeat_dims = [1] * tensor.dim()
    #         repeat_dims[dim] = times
    #         return tensor.repeat(*repeat_dims)

    #     # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
    #     batch_size, l, d = x.size()
    #     # inflated_encoder_feats = _inflate(encoder_feats, beam_width, 0) # ABC --> AABBCC -/-> ABCABC
    #     inflated_encoder_feats = x.unsqueeze(1).permute((1,0,2,3)).repeat((beam_width,1,1,1)).permute((1,0,2,3)).contiguous().view(-1, l, d)

    #     # Initialize the decoder
    #     state = torch.zeros(1, batch_size * beam_width, self.sDim)
    #     pos_index = (torch.Tensor(range(batch_size)) * beam_width).long().view(-1, 1)

    #     # Initialize the scores
    #     sequence_scores = torch.Tensor(batch_size * beam_width, 1)
    #     sequence_scores.fill_(-float('Inf'))
    #     sequence_scores.index_fill_(0, torch.Tensor([i * beam_width for i in range(0, batch_size)]).long(), 0.0)
    #     # sequence_scores.fill_(0.0)

    #     # Initialize the input vector
    #     y_prev = torch.zeros((batch_size * beam_width)).fill_(self.num_classes)

    #     # Store decisions for backtracking
    #     stored_scores          = list()
    #     stored_predecessors    = list()
    #     stored_emitted_symbols = list()

    #     for i in range(self.max_len_labels):    
    #         output, state = self.decoder(inflated_encoder_feats, state, y_prev)
    #         log_softmax_output = F.log_softmax(output, dim=1)

    #         sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
    #         sequence_scores += log_softmax_output
    #         scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_width, dim=1)

    #         # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
    #         y_prev = (candidates % self.num_classes).view(batch_size * beam_width)
    #         sequence_scores = scores.view(batch_size * beam_width, 1)

    #         # Update fields for next timestep
    #         predecessors = (candidates / self.num_classes + pos_index.expand_as(candidates)).view(batch_size * beam_width, 1)
    #         state = state.index_select(1, predecessors.squeeze())

    #         # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
    #         stored_scores.append(sequence_scores.clone())
    #         eos_indices = y_prev.view(-1, 1).eq(eos)
    #         if eos_indices.nonzero().dim() > 0:
    #             sequence_scores.masked_fill_(eos_indices, -float('inf'))

    #         # Cache results for backtracking
    #         stored_predecessors.append(predecessors)
    #         stored_emitted_symbols.append(y_prev)

    #     # Do backtracking to return the optimal values
    #     #====== backtrak ======#
    #     # Initialize return variables given different types
    #     p = list()
    #     l = [[self.max_len_labels] * beam_width for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences

    #     # the last step output of the beams are not sorted
    #     # thus they are sorted here
    #     sorted_score, sorted_idx = stored_scores[-1].view(batch_size, beam_width).topk(beam_width)
    #     # initialize the sequence scores with the sorted last step beam scores
    #     s = sorted_score.clone()

    #     batch_eos_found = [0] * batch_size  # the number of EOS found
    #                                         # in the backward loop below for each batch
    #     t = self.max_len_labels - 1
    #     # initialize the back pointer with the sorted order of the last step beams.
    #     # add pos_index for indexing variable with b*k as the first dimension.
    #     t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * beam_width)
    #     while t >= 0:
    #         # Re-order the variables with the back pointer
    #         current_symbol = stored_emitted_symbols[t].index_select(0, t_predecessors)
    #         t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()
    #         eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
    #         if eos_indices.dim() > 0:
    #             for i in range(eos_indices.size(0)-1, -1, -1):
    #                 # Indices of the EOS symbol for both variables
    #                 # with b*k as the first dimension, and b, k for
    #                 # the first two dimensions
    #                 idx = eos_indices[i]
    #                 b_idx = int(idx[0] / beam_width)
    #                 # The indices of the replacing position
    #                 # according to the replacement strategy noted above
    #                 res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
    #                 batch_eos_found[b_idx] += 1
    #                 res_idx = b_idx * beam_width + res_k_idx

    #                 # Replace the old information in return variables
    #                 # with the new ended sequence information
    #                 t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
    #                 current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
    #                 s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
    #                 l[b_idx][res_k_idx] = t + 1

    #         # record the back tracked results
    #         p.append(current_symbol)

    #         t -= 1

    #     # Sort and re-order again as the added ended sequences may change
    #     # the order (very unlikely)
    #     s, re_sorted_idx = s.topk(beam_width)
    #     for b_idx in range(batch_size):
    #         l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

    #     re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size*beam_width)

    #     # Reverse the sequences and re-order at the same time
    #     # It is reversed because the backtracking happens in reverse time order
    #     p = [step.index_select(0, re_sorted_idx).view(batch_size, beam_width, -1) for step in reversed(p)]
    #     p = torch.cat(p, -1)[:,0,:]
    #     return p, torch.ones_like(p)