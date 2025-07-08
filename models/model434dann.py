import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.autograd import Function

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5,self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16*25*22, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        # x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))
        x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x

# 梯度反转层
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainClassifier(nn.Module):
    def __init__(self,feature_num, domain_num):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(100, domain_num),
            nn.LogSoftmax(dim=1),
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DannModel(nn.Module):
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
        super(DannModel, self).__init__()
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
        
        self.embed = LeNet5()
        self.bigru = nn.GRU(self.input_size, self.Hidden_size, self.num_layer, batch_first=True, bidirectional=self.directions)
        self.biout = nn.Linear(in_features=self.Hidden_size*self.num_directions, out_features=self.output_size)

        self.domainclassifier = DomainClassifier(feature_num=128, domain_num=2)

    def forward(self, source_inputs, source_lengths, alpha):
        # @_embedding
        bs, ts, cn, h, w = source_inputs.shape

        # 源域
        embed_inputs = source_inputs.contiguous().view(bs*ts, cn, h, w)
        embed_outputs = self.embed(embed_inputs)

        domain_inputs = ReverseLayerF.apply(embed_outputs, alpha)
        domain_outputs = self.domainclassifier(domain_inputs)

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
        outputs = self.biout(fc_inputs)
        outputs = outputs.contiguous().view(bs, ts, -1)
        return outputs, hiddens, domain_outputs

