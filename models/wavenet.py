""" borrow from https://github.com/Sytronik/denoising-wavenet-pytorch/blob/master/model/dwavenet.py.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

class ResidualConv1d(nn.Module):
    def __init__(
        self,
        residual_channels, 
        gate_channels, 
        kernel_size,
        skip_out_channels=None,
        dropout=1 - 0.95, 
        padding=None, 
        dilation=1,
        bias=True
    ) -> None:
        super(ResidualConv1d, self).__init__()
        
        self.dropout = dropout
        if skip_out_channels is None:
            skip_out_channels = residual_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation   # For non causal dilated convolution

        self.tanhConv = nn.Conv1d(residual_channels, gate_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              bias=bias)

        self.gateConv = nn.Conv1d(residual_channels, gate_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              bias=bias)


        self.conv1x1_out = nn.Conv1d(gate_channels, residual_channels, 1, bias=bias)
        self.conv1x1_skip = nn.Conv1d(gate_channels, skip_out_channels, 1, bias=bias)
    
    def forward(self, x):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
            g (Tensor): B x C x T, Expanded global conditioning features
        Returns:
            Tensor: output
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        tanhx = self.tanhConv(x)
        gatex = self.gateConv(x)

        x = torch.tanh(tanhx) * torch.sigmoid(gatex)

        # For skip connection
        s = self.conv1x1_skip(x)

        # For residual connection
        x = self.conv1x1_out(x)

        # x = (x + residual) * math.sqrt(0.5) #  / sqrt(2) ??
        x = (x + residual)
        return (x, s)


# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels=1, bias=False,
#                  num_layers=20, num_stacks=2, kernel_size=3,
#                  residual_channels=128, gate_channels=128, skip_out_channels=128):
#         super(Encoder, self).__init__()
#         assert num_layers % num_stacks == 0
#         num_layers_per_stack = num_layers // num_stacks
#         # in_channels is 1 for RAW waveform otherwise quantize classes
#         self.first_conv = nn.Conv1d(in_channels, residual_channels, 3, padding=1, bias=bias)

#         self.conv_layers = nn.ModuleList()
#         for n_layer in range(num_layers):
#             dilation = 2**(n_layer % num_layers_per_stack)
#             conv = ResidualConv1d(
#                 residual_channels, gate_channels,
#                 skip_out_channels=skip_out_channels,
#                 kernel_size=kernel_size,
#                 bias=bias,
#                 dilation=dilation,
#                 dropout=1 - 0.95,
#             )
#             self.conv_layers.append(conv)

#         self.last_conv_layers = nn.Sequential(
#             nn.ReLU(True),
#             nn.Conv1d(skip_out_channels, skip_out_channels, 1, bias=True),
#             nn.ReLU(True),
#             nn.Conv1d(skip_out_channels, out_channels, 1, bias=True),
#         )
    
#     def forward(self, x):
#         x = self.first_conv(x)
#         skips = 0
#         for conv in self.conv_layers:
#             x, h = conv(x)
#             skips += h

#         skips *= math.sqrt(1.0 / len(self.conv_layers))
#         x = skips
#         x = self.last_conv_layers(x)
#         return x

class Encoder(nn.Module):
    def __init__(self,in_channels, out_channels=1, bias=False,
                 dilate_rate=[1,2,5,2,2,5], kernel_size=5,
                 residual_channels=128, gate_channels=128, skip_out_channels=128) -> None:
        super(Encoder, self).__init__()
        
        self.first_conv = nn.Conv1d(in_channels, residual_channels, 1, padding=0, bias=bias)
        self.conv_layers = nn.ModuleList()
        num_layers = len(dilate_rate)
        for n_layer in range(num_layers):
            dilation = dilate_rate[n_layer]
            conv = ResidualConv1d(
                residual_channels, gate_channels,
                skip_out_channels=skip_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                dropout=1 - 0.95,
            )
            self.conv_layers.append(conv)

        # self.last_conv_layers = nn.Sequential(
        #     nn.ReLU(True),
        #     nn.Conv1d(skip_out_channels, skip_out_channels, 1, bias=True),
        #     nn.ReLU(True),
        #     nn.Conv1d(skip_out_channels, out_channels, 1, bias=True),
        # )


    def forward(self, x):
        x = self.first_conv(x)
        skips = 0
        for conv in self.conv_layers:
            x, h = conv(x)
            skips += h

        # skips *= math.sqrt(1.0 / len(self.conv_layers))
        x = skips
        # x = self.last_conv_layers(x)
        return x

# if __name__ == '__main__':
#     model = Encoder(in_channels=64, out_channels=128)
#     x = torch.ones([16, 64, 10])
#     y = model(x)
#     print("Shape of y", y.shape)
