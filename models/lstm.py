import torch
from torch import nn
import torch.nn.functional as F

class Lstm(nn.Module):
    def __init__(
        self, 
        input_size=128, 
        num_layer=2, 
        Hidden_size=512, 
        directions=False, 
        output_size=27
    ) -> None:
        super(Lstm, self).__init__()
        self.input_size = input_size                            # default = 128
        self.num_layer = num_layer                              # default = 2
        self.Hidden_size = Hidden_size                          # default = 512
        self.directions = directions                            # default = False  
        self.output_size = output_size                          # default = 27

        if self.directions is True:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.lstm = nn.LSTM(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        self.fc = nn.Linear(self.num_directions * self.Hidden_size, self.output_size)
    
    def forward(self, input_seq, lengths):
        b, s, l = input_seq.shape

        # packing
        input_seq_packed = nn.utils.rnn.pack_padded_sequence(input_seq, lengths=lengths, batch_first=True, enforce_sorted=False)

        # 对输入lstm
        output_packed,(_, _) = self.lstm(input_seq_packed, None)
        
        # unpacking 
        output_packed,len_unpacked = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        # fc
        # output_packed = output_packed.contiguous().view(self.batch_size, s, self.num_directions, self.Hidden_size)
        # output = torch.mean(output_packed, dim=2)
        # output = output.contiguous().view(self.batch_size * s, self.Hidden_size)
        output = output_packed.view(b * s, -1)
        pred = self.fc(output)
        pred = pred.view(b, s, -1)
        return pred