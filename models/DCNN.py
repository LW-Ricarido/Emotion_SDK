import torch
import torch.nn as nn
import torch.nn.functional as func
from copy import deepcopy


class Flatten(nn.Module):
    def forward(self, input):
        N, C, H, W = input.size()
        return input.view(N,-1)

class DLP_Loss(nn.Module):
    def __init__(self,k=20,lam=50):
        super(DLP_Loss, self).__init__()
        self.k = k
        self.lam = lam

    def forward(self,feture,scores,target):
        '''

        :param input: tensor shape(N,C)  FC layer scores
        :param target: tensor N
        :return:
        '''

        # softmax loss
        loss = func.cross_entropy(scores,target)
        N = feture.shape[0]
        # locality preserving loss
        for i in range(N):
            nums = self.kNN(i,feture,target)
            for j in range(len(nums)):
                loss += self.lam * 0.5 * func.mse_loss(feture[i],1 / len(nums) * feture[nums[j]],size_average=False)
        return loss

    def kNN(self,n,input,target):
        dict = {}
        tmp = input.shape[1]
        length = len(target)
        for i in range(length):
            if n != i and target[n] == target[i]:
                dist = func.pairwise_distance(input[n].view(tmp,-1),input[i].view(tmp,-1)).sum()
                dict[i] = dist
        dict = sorted(dict.items(),key=lambda item:item[1])
        nums = []
        for i in range(len(dict)):
            if i < self.k:
                nums.append(dict[i][0])
            else:
                return nums
        return nums

class DLP_CNN(nn.Module):
    def __init__(self,output_classes):
        super(DLP_CNN,self).__init__()
        self.conv1 =  nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,96,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(36864,2000)
        self.fc2_ = nn.Linear(2000,output_classes)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv2(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv3(input)
        input = self.relu(input)
        input = self.conv4(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv5(input)
        input = self.relu(input)
        input = self.conv6(input)
        input = self.relu(input)
        input = self.flatten(input)
        input = self.fc1(input)
        input = self.relu(input)
        feature = input
        input = self.fc2_(input)
        return input,feature



def get_DLP_CNN(output_classes,pretrain=False):
    if pretrain:
        model = DLP_CNN(output_classes)
        pretrained_dict = torch.load('../pretrain/origin_pic_model.pth')
        model_dict = model.state_dict()

        keys = deepcopy(pretrained_dict).keys()

        for key in keys:
            if key not in model_dict:
                del pretrained_dict[key]

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
    return DLP_CNN(output_classes)
