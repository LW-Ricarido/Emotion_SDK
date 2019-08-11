# Emotion SDK

This emotion sdk based on [PyTorch](/[https:/pytorch.org](https:/pytorch.org/)) could be used for both video and image face emotion recognition.

<!-- TOC -->autoauto- [Emotion SDK](#emotion-sdk)auto    - [Prerequistes](#prerequistes)auto    - [Usage](#usage)auto        - [Parameters:](#parameters)auto        - [Method](#method)auto            - [train()](#train)auto            - [eval()](#eval)auto            - [recognize()](#recognize)auto            - [batch_recognize()](#batch_recognize)auto    - [Dataset](#dataset)auto        - [Image](#image)auto        - [Video](#video)auto    - [Model](#model)auto        - [Pretrain](#pretrain)autoauto<!-- /TOC -->

## Prerequistes

1. Python 3.6 or greater
2. PyTorch 1.0 or greater
3. CUDA 9.0 or greater(Using GPU)
4. TensorboardX

## Usage

To use this emotion sdk, you only need to import class `Emotion_Recognize`. 

### Parameters:

- Video(`bool`,optional) - Controls recognize video or image. Default:`True`
- pretrain(`bool`,optional)  - Using pretrained model or not. Default:`True`
- nGPU(`int`,optional) - Number of gpus to use. Default:0 

### Method

#### train()

To train or fine-tune `Emotion_Recognize` on yor own datasets.

`train` method will call eval method after every epoch to evaluate model on test dataset. And the method eventually return a dictionary(`Summary{'acc':,'acc_top3','best_epoch':}`) which contains the best top1 results in test dataset and save the best evaluation model weights in pretrain/best_pic_model.pth or pretrain/best_video_model.pth. Train record will be saved in log dictionary and could be visualized by Tensorboard.

- train_loader(`dataloader`) - Dataloder for train dataset. 
- test_loader(`dataloader`) - Dataloader for test dataset.
- epochs(`int`) - Number of train epoch
- learn_rate(`float`,optional) - learning rate. Default:0.00003
- momentum(`float`,optional) - momentum factor. Default:0.9
- weight_deacy(`float`,optional) - weight decay. Default:0.0005

#### eval()

To evaluate model on test dataset.

`eval()` method will return a dictionary(`Summary{'acc':,'acc_top3'}`) which contains the results of this evaluation.

- test_loader(`dataloader`) - Dataloader for test dataset.
- epoch(`int`,optional) - The number of epoch which is most used for record training result when `train()` method call `eval()`. Default:1

#### recognize()

To recognize single image or video clip.

`recognize()` will return a int.

- input(`tensor`) - Image or video clip data\

#### batch_recognize()

To recognize image or video clip in batchs. And this method will return a dictionary(`results{'filename':prediction}`)

- input_loader:(`dataloader`) - Dataloader for input dataset

And this input_loader, should change Dataset from `image/video, target` to `image/video, filename`. 

## Dataset

### Image

For image dataset, the input image tensor's size should be [3,100,100], and image label should be a number from 0 to 6. And there is a image training dataset example:

```python
class RAFTrainSet(data.Dataset):
    def __init__(self,args):
        self.images = list()
        self.targets = list()
        self.args = args
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
```

### Video

For video dataset, every video clip should contain **16** images and every image size should be [3,224,224]. Video label should be a number from 0 to 6. And there is a video training dataset example.

```python
class AFEWTrainSet(data.Dataset):
    def __init__(self,args):
        self.images = list()
        self.targets = list()
        self.args = args

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
```

## Model

The image model was based on this paper -- [Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf). And my model achieves 75.489% acc on RAF_DB.

The video model was based on this paper -- [Video-Based Emotion Recognition using CNN-RNN and C3D Hybrid Networks](https://dl.acm.org/citation.cfm?id=2997632). And my model achieves 41% acc on AFEW.

### Pretrain

To use pretrain model, you should name pretrain model as `origin_pic_model.pth` or `origin_video_model.pth` and put in it in pretrain dictionary.

[Video pretrain model](https://drive.google.com/open?id=1RzGe5pDbcaQQtQE8h7fI9FUGdAd0BgzP)

