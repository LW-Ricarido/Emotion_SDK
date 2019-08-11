import torch
import torch.nn as nn
from copy import deepcopy
from models.VGG import VggFace

class Flatten(nn.Module):
    def forward(self, input):
        S, N, F = input.size()
        return input.view(N,-1)


class CNN_LSTM(nn.Module):
    def __init__(self,output_classes=7):
        super(CNN_LSTM, self).__init__()
        self.vgg = VggFace()
        self.lstm = nn.LSTM(input_size=4096, hidden_size=128 ,num_layers=1, bidirectional=True)
        self.fc = nn.Linear(128 * 16 * 2, output_classes)
        self.flatten = Flatten()


    def forward(self, input):
        x = []
        x.append(self.vgg(input[:, :, 0, :]))
        x.append(self.vgg(input[:, :, 1, :]))
        x.append(self.vgg(input[:, :, 2, :]))
        x.append(self.vgg(input[:, :, 3, :]))
        x.append(self.vgg(input[:, :, 4, :]))
        x.append(self.vgg(input[:, :, 5, :]))
        x.append(self.vgg(input[:, :, 6, :]))
        x.append(self.vgg(input[:, :, 7, :]))
        x.append(self.vgg(input[:, :, 8, :]))
        x.append(self.vgg(input[:, :, 9, :]))
        x.append(self.vgg(input[:, :, 10, :]))
        x.append(self.vgg(input[:, :, 11, :]))
        x.append(self.vgg(input[:, :, 12, :]))
        x.append(self.vgg(input[:, :, 13, :]))
        x.append(self.vgg(input[:, :, 14, :]))
        x.append(self.vgg(input[:, :, 15, :]))
        x = torch.stack(x)
        x = self.lstm(x)[0]
        x = self.flatten(x)
        return x

def get_CNN_LSTM(output_classes=7):
    return CNN_LSTM(output_classes=output_classes)