#==============================================
# Lenet 5 model
# Data : 2022 / 4 / 22
# costumized : setting_parameter
#==============================================

import string
import torch.nn as nn
import torch.nn.functional as F


# class LeNet5(nn.Module):
#     def __init__(self) -> None:
#         super(LeNet5,self).__init__()

#         self.conv1 = nn.Conv2d(1, 8, 5)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
#         self.conv2 = nn.Conv2d(8, 16, 5)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.batchnorm2 = nn.BatchNorm2d(num_features=16)

#         self.fc1 = nn.Linear(16*4*4, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)

#         self.dropout = nn.Dropout(0.1)
    
#     def forward(self,x):
#         # x = self.pool1(F.relu(self.conv1(x)))
#         x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))
#         x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))
#         # x = self.pool2(F.relu(self.conv2(x)))
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.dropout(x)
#         return x

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

class LeNet6(nn.Module):
    def __init__(self) -> None:
        super(LeNet6,self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)


        self.fc1 = nn.Linear(32*10*9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        # x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))
        x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.pool3(self.conv3(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x

class LeNet7(nn.Module):
    def __init__(self) -> None:
        super(LeNet7,self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)

        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batchnorm4 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(64*3*2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))
        x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.pool3(self.conv3(x))))
        x = F.relu(self.batchnorm4(self.pool4(self.conv4(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x


class LeNet8(nn.Module):
    def __init__(self) -> None:
        super(LeNet8,self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=8)
        
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.batchnorm2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)

        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.batchnorm4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(128*1*1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x = F.relu(self.batchnorm1(self.pool1(self.conv1(x))))
        x = F.relu(self.batchnorm2(self.pool2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.pool3(self.conv3(x))))
        x = F.relu(self.batchnorm4(self.pool4(self.conv4(x))))
        x = F.relu(self.batchnorm5(self.pool5(self.conv5(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dropout(x)
        return x
# class LeNet5(nn.Module):
#     def __init__(self) -> None:
#         super(LeNet5,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 8, 5),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),

#             nn.Conv2d(8, 16, 5),
#             nn.MaxPool2d(2, 2),
#             nn.BatchNorm2d(16),
#             nn.ReLU()
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(16*4*4, 512),
#             nn.ReLU(),

#             nn.Linear(512, 256),
#             nn.ReLU(),

#             nn.Linear(256, 128)
#         )

#         self.dropout = nn.Dropout(0.1)

#     def forward(self,x):
#         x = self.conv(x)
#         x = x.view(x.shape[0], -1)
#         x = self.fc(x)
#         x = self.dropout(x)
#         return x

    
#     def setting_parameter(self, in_channel, out_channel, input_shape, dropout) -> None:
#         h = int((input_shape[0] - 4)/2)
#         h = int((h - 4) / 2)
#         w = int((input_shape[1] - 4)/2)
#         w = int((w - 4) / 2)

#         self.fc1 = nn.Linear(in_features=(16 * h * w), out_features=512)
#         self.fc3 = nn.Linear(in_features=256, out_features=out_channel)
#         self.dropout = nn.Dropout(dropout)