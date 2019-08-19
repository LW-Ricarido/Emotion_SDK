import os
import cv2
import random
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image




def get_Img_train_loader(batch_size,shuffle=True,workers=0):
    dataset = RAFTrainSet(train_list='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/train_set',
                          data_dir='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/Image/aligned')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )

def get_Img_test_loader(batch_size,shuffle=True,workers=0):
    dataset = RAFTestSet(test_list='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/validation_set',
                         data_dir='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/Image/aligned')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )

def get_Img_recog_loader(batch_size,shuffle=True,workers=0):
    dataset = RAFRecogSet(test_list='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/test_set',
                          data_dir='/Users/liwei/Documents/GitHub/Emotion-Recognize/DataSet/RAF/basic/Image/aligned')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )

def get_Video_train_loader(batch_size,shuffle=True,workers=0):
    dataset = AFEWTrainSet(train_list='/Users/liwei/Documents/GitHub/Video-Facial-Emotion-Recognition/datalist/train_set.txt',
                           train_data_dir='/Users/liwei/Computer Vision/计算机视觉论文/Emotion/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/Faces')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )

def get_Video_test_loader(batch_size,shuffle=True,workers=0):
    dataset = AFEWTestSet(test_list='/Users/liwei/Documents/GitHub/Video-Facial-Emotion-Recognition/datalist/val_set.txt',
                          test_data_dir='/Users/liwei/Computer Vision/计算机视觉论文/Emotion/AFEW/Val_AFEW/AlignedFaces_LBPTOP_Points_Val/Faces')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )

def get_Video_recog_loader(batch_size,shuffle=True,workers=0):
    dataset = AFEWRecogSet(test_list='/Users/liwei/Documents/GitHub/Video-Facial-Emotion-Recognition/datalist/val_set.txt',
                          test_data_dir='/Users/liwei/Computer Vision/计算机视觉论文/Emotion/AFEW/Val_AFEW/AlignedFaces_LBPTOP_Points_Val/Faces')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True
    )


class RAFTrainSet(data.Dataset):
    def __init__(self,train_list,data_dir):
        self.images = list()
        self.targets = list()

        lines = open(train_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(data_dir,path))
            self.targets.append(int(label) - 1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(100,100))
        image = self.transform(image)
        target = self.targets[index]
        return image,target


    def __len__(self):
        return len(self.targets)

class RAFTestSet(data.Dataset):
    def __init__(self,data_dir,test_list):
        self.images = list()
        self.targets = list()
        lines = open(test_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(data_dir, path))
            self.targets.append(int(label) - 1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(100,100))
        image = self.transform(image)
        target = self.targets[index]
        return image,target

    def __len__(self):
        return len(self.targets)

class RAFRecogSet(data.Dataset):
    def __init__(self,test_list,data_dir):
        self.images = list()
        self.targets = list()

        lines = open(test_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(data_dir, path))
            self.targets.append(int(label) - 1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(100,100))
        image = self.transform(image)
        target = self.targets[index]
        return image,self.images[index]

    def __len__(self):
        return len(self.images)

class AFEWTrainSet(data.Dataset):
    def __init__(self,train_list,train_data_dir):
        self.images = list()
        self.targets = list()

        lines = open(train_list).readlines()
        for lines in lines:
            path, label = lines.strip().split(' ')
            self.images.append(os.path.join(train_data_dir,path))
            self.targets.append(int(label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imglist = os.listdir(self.images[index])
        imglist.sort()
        lists = random.sample(range(0,imglist.__len__()),16)
        lists.sort()
        video = []
        for i in lists:
            image = Image.open(self.images[index]+'/'+imglist[i])
            video.append(self.transform(image))
        video = torch.stack(video)
        D, C, H, W = video.size()
        video = video.reshape(C, D, H, W)
        target = self.targets[index]
        return video,target

    def __len__(self):
        return len(self.targets)

class AFEWTestSet(data.Dataset):
    def __init__(self,test_list,test_data_dir):
        self.images = list()
        self.targets = list()

        lines = open(test_list).readlines()
        for lines in lines:
            path, label = lines.strip().split(' ')
            self.images.append(os.path.join(test_data_dir,path))
            self.targets.append(int(label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imglist = os.listdir(self.images[index])
        imglist.sort()
        lists = random.sample(range(0,imglist.__len__()),16)
        lists.sort()
        video = []
        for i in lists:
            image = Image.open(self.images[index]+'/'+imglist[i])
            video.append(self.transform(image))
        video = torch.stack(video)
        D, C, H, W = video.size()
        video = video.reshape(C,D,H,W)
        target = self.targets[index]
        return video,target

    def __len__(self):
        return len(self.targets)

class AFEWRecogSet(data.Dataset):
    def __init__(self,test_list,test_data_dir):
        self.images = list()
        self.targets = list()

        lines = open(test_list).readlines()
        for lines in lines:
            path, label = lines.strip().split(' ')
            self.images.append(os.path.join(test_data_dir,path))
            self.targets.append(int(label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imglist = os.listdir(self.images[index])
        imglist.sort()
        lists = random.sample(range(0,imglist.__len__()),16)
        lists.sort()
        video = []
        for i in lists:
            image = Image.open(self.images[index]+'/'+imglist[i])
            video.append(self.transform(image))
        video = torch.stack(video)
        D, C, H, W = video.size()
        video = video.reshape(C,D,H,W)
        target = self.targets[index]
        return video,self.images[index]

    def __len__(self):
        return len(self.targets)