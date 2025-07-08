from . import Lenet5, lstm

import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet_lstm(nn.Module):
    def __init__(
        self, 
        cnn_input_nc, 
        cnn_output_features, 
        cnn_input_shape, 
        lstm_num_layer, 
        lstm_hidden_size, 
        lstm_directions, 
        lstm_output_size
    ) -> None:
        super(Lenet_lstm, self).__init__()

        # CNN 模型的参数
        self.cnn_input_nc = cnn_input_nc                                        # cnn input channel num 【default = 3】 
        self.cnn_output_features = cnn_output_features                          # cnn output feature num 【default = 128】
        self.cnn_input_shape = cnn_input_shape                                  # cnn input shape

        # LSTM模型的参数
        self.lstm_input_size = cnn_output_features                              # lstm input size
        self.lstm_num_layer = lstm_num_layer                                    # lstm layer num
        self.lstm_hidden_size = lstm_hidden_size                                # lstm hidden size 【default = 512】 
        self.lstm_directions = lstm_directions
        self.lstm_output_size = lstm_output_size

        # 生成 cnn 模型
        self.cnn = Lenet5.LeNet5()              # 创建原始lenet5模型

        self.cnn.setting_parameter(
            in_channel=self.cnn_input_nc, 
            out_channel=self.cnn_output_features, 
            input_shape=self.cnn_input_shape
        )

        
        self.lstm = lstm.Lstm(
            input_size=self.lstm_input_size, 
            num_layer=self.lstm_num_layer,
            Hidden_size=self.lstm_hidden_size,
            directions=self.lstm_directions,
            output_size=self.lstm_output_size
        )
    
    def forward(self, inputs, lengths):
        batch_size, timesteps, channel_x, h_x, w_x = inputs.shape
        conv_input = inputs.view(batch_size * timesteps, channel_x, h_x, w_x)
        conv_output = self.cnn(conv_input)  # 【b*ts, ip_s】

        lstm_input = conv_output.view(batch_size, timesteps, -1)
        lstm_output = self.lstm(lstm_input, lengths) # 【b, ts, cln】
        
        return lstm_output