import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(16*4*4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.1)

    def forward(self,x):
        x = self.conv1(x)                   # b, 1, 110, 96
        x = self.pool1(x)                   
        x = F.relu(self.batchnorm1(x))      

        x = self.conv2(x)                   # b, 8, 55, 48
        x = self.pool2(x)                   # b, 16, 51, 44
        x = F.relu(self.batchnorm2(x))      # b, 16, 25, 22

        feature = x.view(x.shape[0], -1)

        feature = F.relu(self.fc1(feature))
        feature = F.relu(self.fc2(feature))
        feature = F.relu(self.fc3(feature))
        feature = self.dropout(feature)

        return feature

    def setting_parameter(self, in_channel, out_channel, input_shape, dropout):
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=5)
        h = int((input_shape[0] - 4)/2)
        h = int((h - 4) / 2)
        w = int((input_shape[1] - 4)/2)
        w = int((w - 4) / 2)

        self.fc1 = nn.Linear(in_features=(16 * h * w), out_features=512)
        self.fc3 = nn.Linear(in_features=256, out_features=out_channel)
        self.dropout = nn.Dropout(dropout)

        return h, w

class Decoder(nn.Module):
    def __init__(self, outsize, h, w) -> None:
        super(Decoder, self).__init__()

        self.outsize = outsize
        self.h = h
        self.w = w

        self.upfc1 = nn.Linear(self.outsize, 256)
        self.upfc2 = nn.Linear(256, 512)
        self.upfc3 = nn.Linear(512, 16*self.h*self.w)

        self.upsample1 = nn.Upsample(size=(51, 44), mode="bilinear", align_corners=True)

        self.tconv1 = nn.ConvTranspose2d(16, 8, 5)

        self.upsample2 = nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True)
        self.tconv2 = nn.ConvTranspose2d(8, 1, 5)
    
    def forward(self, feature):
        feature = F.relu(self.upfc1(feature))
        feature = F.relu(self.upfc2(feature))
        feature = F.relu(self.upfc3(feature))

        x = feature.view(feature.shape[0], 16, self.h, self.w)
        x = self.upsample1(x)
        x = self.tconv1(x)
        x = self.upsample2(x)
        x = self.tconv2(x)
        return x


class AlexNet(nn.Module):

    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, input_shape, dropout) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        h, w = self.encoder.setting_parameter(in_channel, out_channel, input_shape, dropout)

        # self.encoder = AlexNet()

        # self.encoder = tmodels.resnet18()
        # self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # self.encoder.fc = nn.Linear(in_features=512, out_features=128, bias=True)

        # self.encoder = tmodels.mobilenet_v3_large()
        # self.encoder.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.encoder.classifier[3] = nn.Linear(1280, 128, bias=True)

        # self.encoder = tmodels.vgg16()
        # self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.encoder.classifier[6] = nn.Linear(4096, 128, bias=True)
        
        # h = 25
        # w = 22
        self.decoder = Decoder(out_channel, h, w)

    def forward(self, x):
        feature = self.encoder(x)
        x_ = self.decoder(feature)

        return feature, x_



# net = AutoEncoder(1, 128, (114, 100), 0.3)
# inputs = torch.randn(size=(16, 1, 114, 100), dtype=torch.float32)
# feature, out = net(inputs)
# print(feature.shape)
# print(out.shape)        



# class Encoder(nn.Module):
#     def __init__(self) -> None:
#         super(Encoder,self).__init__()
#         self.conv1 = nn.Conv2d(1, 8, 3)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
#         self.conv2 = nn.Conv2d(8, 16, 3)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.batchnorm2 = nn.BatchNorm2d(num_features=16)

#         self.conv3 = nn.Conv2d(16, 32, 3)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        

#     def forward(self,x):
#         x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))    

#         x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))

#         feature = F.relu(self.batchnorm3(self.pool3(self.conv3(x))))
#         return feature

# class Decoder(nn.Module):
#     def __init__(self) -> None:
#         super(Decoder, self).__init__()

#         self.upsample1 = nn.Upsample(size=(25, 21), mode="bilinear", align_corners=True)
#         self.tconv1 = nn.ConvTranspose2d(32, 16, 3)

#         self.upsample2 = nn.Upsample(size=(54, 47), mode="bilinear", align_corners=True)
#         self.tconv2 = nn.ConvTranspose2d(16, 8, 3)

#         self.upsample3 = nn.Upsample(size=(112, 98), mode='bilinear', align_corners=True)
#         self.tconv3 = nn.ConvTranspose2d(8, 1, 3)
    
#     def forward(self, feature):
#         x = self.upsample1(feature)
#         x = self.tconv1(x)

#         x = self.upsample2(x)
#         x = self.tconv2(x)

#         x = self.upsample3(x)
#         x = self.tconv3(x)

#         return x

# class AutoEncoder(nn.Module):
#     def __init__(self) -> None:
#         super(AutoEncoder, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         feature = self.encoder(x)
#         x_ = self.decoder(feature)

#         return feature, x_

# net = AutoEncoder()
# print(net)

# inputs = torch.randn(size=(16, 1, 114, 100))
# _,outs = net(inputs)
# print(outs.shape)



# class Encoder(nn.Module):
#     def __init__(self) -> None:
#         super(Encoder,self).__init__()
#         self.layer1 = nn.Linear(114*100, 4096)
#         self.layer2 = nn.Linear(4096, 2048)
#         self.layer3 = nn.Linear(2048, 512)
#         self.layer4 = nn.Linear(512, 128)
        

#     def forward(self,x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         feature = F.relu(self.layer4(x))
#         return feature

# class Decoder(nn.Module):
#     def __init__(self) -> None:
#         super(Decoder, self).__init__()
#         self.tlayer1 = nn.Linear(128, 512)
#         self.tlayer2 = nn.Linear(512, 2048)
#         self.tlayer3 = nn.Linear(2048, 4096)
#         self.tlayer4 = nn.Linear(4096, 114*100)
    
#     def forward(self, feature):
#         x = F.relu(self.tlayer1(feature))
#         x = F.relu(self.tlayer2(x))
#         x = F.relu(self.tlayer3(x))
#         x = F.relu(self.tlayer4(x))
#         return x

# class AutoEncoder(nn.Module):
#     def __init__(self) -> None:
#         super(AutoEncoder, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         b, c, h, w = x.shape
#         x = x.view(b, -1)
#         feature = self.encoder(x)
#         x_ = self.decoder(feature)

#         x_ = x_.view(b, c, h, w)

#         return feature, x_

# net = AutoEncoder()
# print(net)

# inputs = torch.randn(size=(16, 1, 114, 100))
# _,out = net(inputs)
# print(out.shape)