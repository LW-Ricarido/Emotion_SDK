import torch
import torch.nn as nn
from copy import deepcopy


class Flatten(nn.Module):
    def forward(self, input):
        N, C, D, H, W = input.size()
        return input.view(N,-1)

class C3DNetwork(nn.Module):
    def __init__(self,output_classes=7):
        super(C3DNetwork, self).__init__()
        self.conv1 = nn.Conv3d(3,64,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
        self.conv2 = nn.Conv3d(64,128,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.conv3a = nn.Conv3d(128,256,kernel_size=3,padding=1)
        self.conv3b = nn.Conv3d(256,256,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.conv4a = nn.Conv3d(256,512,kernel_size=3,padding=1)
        self.conv4b = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.conv5a = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.conv5b = nn.Conv3d(512,512,kernel_size=3,padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.flatten = Flatten()
        self.fc6 = nn.Linear(512*int(224/32)*int(224/32),4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,output_classes)
        self.dropout = nn.Dropout()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.relu(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


def get_C3D(output_classes=7):
    return C3DNetwork(output_classes=output_classes)
