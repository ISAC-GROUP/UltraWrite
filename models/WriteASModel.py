import torch
import torch.nn as nn
from . import wavenet
from . import Lenet5
from . import AttenDecoder
# import wavenet, Lenet5, AttenDecoder


class WriteAsModel(nn.Module):
    def __init__(
        self,
        front_end_in_channel=1,
        front_end_out_channel=128,
        front_end_input_shape=(114, 100),
        front_end_dropout=0.3,
    
        encoder_out_channel=128,
        # encoder_num_layers=20,
        # encoder_num_stacks = 2,
        encoder_dilate_rate=[1,2,5,2,2,5],
        encoder_kernel_size = 3,
        encoder_residual_channels = 128,
        encoder_gate_channels = 128,
        encoder_skip_out_channels=128,
        encoder_fc_out_size=27,
        
        decoder_input_size=128,
        decoder_hidden_size=128, 
        decoder_num_layer=2,
        decoder_direction=False,
        decoder_num_embed=28,
        decoder_embed_dim=128,
        decoder_dropout=0.0,
        decoder_output_size=27,
        decoder_atten_mode='gereral',
        decoder_pad=27,
        BOS_Token=27,
        EOS_Token=26
    ) -> None:
        super(WriteAsModel, self).__init__()
        #=====================================================
        # front end model
        self.front_end_in_channel = front_end_in_channel
        self.front_end_out_channel = front_end_out_channel
        self.front_end_input_shape = front_end_input_shape
        self.front_end_dropout = front_end_dropout

        self.front_end_net = Lenet5.LeNet5()
        self.front_end_net.setting_parameter(
            in_channel=self.front_end_in_channel, 
            out_channel=self.front_end_out_channel, 
            input_shape=self.front_end_input_shape,
            dropout=self.front_end_dropout
        )

        #=====================================================
        # encoder model
        self.encoder_in_channel = front_end_out_channel
        self.encoder_out_channel = encoder_out_channel
        # self.encoder_num_layers = encoder_num_layers
        # self.encoder_num_stacks = encoder_num_stacks
        self.encoder_dilate_rate = encoder_dilate_rate
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_residual_channels = encoder_residual_channels
        self.encoder_gate_channels = encoder_gate_channels
        self.encoder_skip_out_channels = encoder_skip_out_channels
        self.encoder_fc_out_size = encoder_fc_out_size

        self.encoder = wavenet.Encoder(
            in_channels=self.encoder_in_channel,
            out_channels=self.encoder_out_channel,
            # num_layers=self.encoder_num_layers,
            # num_stacks=self.encoder_num_stacks,
            dilate_rate=self.encoder_dilate_rate,
            kernel_size=self.encoder_kernel_size,
            residual_channels=self.encoder_residual_channels,
            gate_channels=self.encoder_gate_channels,
            skip_out_channels=self.encoder_skip_out_channels
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=self.encoder_skip_out_channels, out_features=self.encoder_fc_out_size)
            # nn.Linear(in_features=128, out_features=self.encoder_fc_out_size)
        )

        # =====================================================
        # decoder model
        self.decoder_input_size=decoder_input_size
        self.decoder_hidden_size=decoder_hidden_size
        self.decoder_num_layer=decoder_num_layer
        self.decoder_direction=decoder_direction
        self.decoder_num_embed=decoder_num_embed
        self.decoder_embed_dim=decoder_embed_dim
        self.decoder_dropout=decoder_dropout
        self.decoder_output_size=decoder_output_size
        self.decoder_atten_mode=decoder_atten_mode
        self.decoder_pad=decoder_pad
        self.decoder_BOS_Token=BOS_Token
        self.decoder_EOS_Token=EOS_Token


        self.decoder = AttenDecoder.Decoder(
            decoder_input_size=self.decoder_input_size,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_num_layer=self.decoder_num_layer,
            decoder_direction=self.decoder_direction,
            decoder_num_embed=self.decoder_num_embed,
            decoder_embed_dim=self.decoder_embed_dim,
            dropout=self.decoder_dropout,
            decoder_output_size=self.decoder_output_size,
            atten_mode=self.decoder_atten_mode,
            pad=self.decoder_pad,
            BOS_Token=self.decoder_BOS_Token,
            EOS_Token=self.decoder_EOS_Token
        )

    def forward(self, source_inputs, source_lengths, target_inputs, target_lengths, max_length):
        '''Forward
        Parameters:
            source_inputs(tensor): B x TS x C x H x W
            source_lengths(tensor): B x 1
            target_inputs(tensor): B x TS
            target_lengths(tensor): B x 1
            max_length(int): 
        '''
        B, TS, C, H, W = source_inputs.shape
        source_inputs = source_inputs.view(B*TS, C, H, W)
        front_end_out = self.front_end_net(source_inputs)                       # [(B*TS) x front_end_out_channel(128)]
        
        encoder_input = front_end_out.contiguous().view(B, TS, -1)              # [B x TS x front_end_out_channel(128)]
        encoder_input = encoder_input.permute(0, 2, 1)                          # [B x front_end_out_channel(128) x TS]
        encoder_output = self.encoder(encoder_input)                            # [B x encoder_out_channel(256) x TS]
        encoder_output = encoder_output.permute(0, 2, 1)                        # [B x TS x encoder_out_channel(256)]

        device = encoder_output.device

        encoder_last_hidden = torch.zeros(size=(self.decoder_num_layer, B, self.decoder_hidden_size))   # [num_layer x B x hidden_size(256)]
        for i in range(B):
            for j in range(self.decoder_num_layer):
                encoder_last_hidden[j][i] = encoder_output[i][source_lengths[i]-1]            

        encoder_last_hidden = encoder_last_hidden.to(device)

        if self.training is True:
            decoder_output = self.decoder(
                trg_inputs=target_inputs,
                trg_len=target_lengths,
                source_len=source_lengths,
                encoder_outputs=encoder_output,
                encoder_last_hidden=encoder_last_hidden,
                max_length=max_length
            )
        elif self.training is False:
            decoder_output = self.decoder(
                trg_inputs = None,
                trg_len = None,
                source_len = source_lengths,
                encoder_outputs = encoder_output,
                encoder_last_hidden=encoder_last_hidden,
                max_length=max_length
            )
        
        encoder_output = encoder_output.contiguous().view(B*TS, -1)
        encoder_fc_output = self.encoder_fc(encoder_output)
        encoder_fc_output = encoder_fc_output.view(B, TS, -1)

        return encoder_fc_output, decoder_output



# if __name__ == '__main__':
#     net = WriteAsModel(
#         front_end_in_channel=1,
#         front_end_out_channel=128,
#         front_end_input_shape=(114, 100),
#         front_end_dropout=0.4,

#         encoder_out_channel=128,
#         # encoder_num_layers=20,
#         # encoder_num_stacks=2,
#         encoder_dilate_rate=[1,2,5,2,2,5],
#         encoder_kernel_size=5,
#         encoder_residual_channels=128,
#         encoder_gate_channels=128,
#         encoder_skip_out_channels=128,
#         encoder_fc_out_size=27,

#         decoder_input_size=128,
#         decoder_hidden_size=128,
#         decoder_num_layer=2,
#         decoder_direction=False,
#         decoder_num_embed=28,
#         decoder_embed_dim=128,
#         decoder_dropout=0.0,
#         decoder_output_size=27,
#         decoder_atten_mode='general',
#         decoder_pad=27,
#         BOS_Token=27,
#         EOS_Token=26
#     )

#     print(net)
#     source_inputs = torch.randn(size=(4, 20, 1, 114, 100), dtype=torch.float32)
#     source_lengths = torch.tensor([10, 8, 7, 5], dtype=torch.int32)
#     target_inputs = torch.randint(low=0, high=27, size=(4, 8))
#     target_lengths = torch.tensor([8, 5, 5, 4])

#     encoder_fc_output, decoder_output = net(source_inputs, source_lengths, target_inputs, target_lengths, 8)
#     print(encoder_fc_output.shape)
#     print(decoder_output.shape)

#     source_inputs = torch.randn(size=(4, 12, 1, 114, 100), dtype=torch.float32)
#     source_lengths = torch.tensor([10, 8, 7, 5], dtype=torch.int32)
#     target_inputs = torch.randint(low=0, high=27, size=(4, 8))
#     target_lengths = torch.tensor([8, 5, 5, 4])
#     encoder_fc_output, decoder_output = net(source_inputs, source_lengths, target_inputs, target_lengths, 8)
#     print(encoder_fc_output.shape)
#     print(decoder_output.shape)
