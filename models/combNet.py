import torch
import torch.nn as nn
from copy import deepcopy
from models.CNN_RNN import get_CNN_LSTM
from models.C3D import get_C3D

class Fusion(nn.Module):
    def __init__(self,lam = 1):
        super(Fusion, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam), requires_grad=True)

    def forward(self, x, y):
        return self.lam * x + y


class combNetwork(nn.Module):
    def __init__(self,output_classes=7):
        super(combNetwork, self).__init__()
        self.c3d = get_C3D(output_classes=output_classes)
        self.cnn_rnn = get_CNN_LSTM(output_classes)
        for param in self.cnn_rnn.parameters():
            param.requires_grad = False
        self.fc_ = nn.Linear(128 * 16 * 2 + 4096, output_classes)

    def forward(self, input):
        x = self.c3d(input)
        y = self.cnn_rnn(input)
        feature = torch.cat((x,y),1)
        out = self.fc_(feature)
        return out

def get_combNetwork(output_classes=7,pretrain=False):
    if pretrain:
        model = combNetwork(output_classes)
        pretrained_dict = torch.load('../pretrain/origin_video_model.pth')
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
    model = combNetwork(output_classes)
    return model
