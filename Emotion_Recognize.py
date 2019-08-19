import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from models.DCNN import get_DLP_CNN,DLP_Loss
from models.combNet import get_combNetwork
from torch.utils.data import dataloader
from tensorboardX import SummaryWriter
class Emotion_Recognize():

    def __init__(self,Video=True,pretrain=True,nGPU=0,emotion_No=7):
        super(Emotion_Recognize, self).__init__()
        self.video = Video
        self.nGPU = nGPU
        if self.video:
            self.model = get_combNetwork(emotion_No,pretrain=pretrain)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.model = get_DLP_CNN(emotion_No,pretrain=pretrain)
            self.criterion = DLP_Loss()
        self.writer = SummaryWriter('log')
        if self.nGPU > 0:
            self.criterion = self.criterion.cuda()
            if self.nGPU > 1:
                self.model = nn.DataParallel(self.model, device_ids=[i for i in range(self.nGPU)]).cuda()
            else:
                self.model = self.model.cuda()

    def train(self,train_loader:dataloader,test_loader:dataloader,epochs:int,learn_rate=0.00003,momentum=0.9, weight_decay=0.0005,decay=30):
        '''

        :param train_loader:
        :param test_loader:
        :param epochs:
        :param learn_rate:
        :param momentum:
        :param weight_decay:
        :return:
        '''
        self.learn_rate = learn_rate
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            learn_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True
        )
        acc_avg = 0
        acc_top3_avg = 0
        loss_avg = 0
        total = 0

        model = self.model
        model.train()
        max_acc = 0
        checkpoint = dict()
        for epoch in range(1,epochs+1):
            self.learning_rate(epoch,decay=decay)

            for i, (input_tensor, target) in enumerate(train_loader):

                if self.nGPU > 0:
                    input_tensor = input_tensor.cuda()
                    target = target.cuda()

                batch_size = target.size(0)

                if self.video:

                    output = model(input_tensor)
                    loss = self.criterion(output, target)

                else:
                    output,feature = model(input_tensor)
                    loss = self.criterion(output,feature,target)

                acc, acc_top3 = self.accuracy(output.data, target, (1,3))

                acc_avg += acc * batch_size
                acc_top3_avg += acc_top3 * batch_size

                loss_avg += loss.item() * batch_size

                total += batch_size

                self.optimizer.zero_grad()
                loss.backward()

            loss_avg /= total
            acc_avg /= total
            acc_top3_avg /= total
            self.writer.add_scalar("Train/Loss", loss_avg, epoch)
            self.writer.add_scalar("Train/Acc", acc_avg, epoch)
            self.writer.add_scalar("Train/Acc_Top3", acc_top3_avg, epoch)
            summary = self.eval(test_loader=test_loader)
            if summary['acc'] > max_acc:
                max_acc = summary['acc']
                summary['best_epoch'] = epoch
                checkpoint['model'] = model.state_dict()

        if self.video:
            torch.save(checkpoint, './pretrain/best_video_model.pth')
        else:
            torch.save(checkpoint, './pretrain/best_pic_model.pth')
        return summary


    def eval(self,test_loader:dataloader,epoch=1):
        acc_avg = 0
        acc_top3_avg = 0
        total = 0
        summary = dict()

        model = self.model
        model.eval()
        for i, (input_tensor, target) in enumerate(test_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda()
            batch_size = target.size(0)

            if self.video:
                output = model(input_tensor)
            else:
                output,feature = model(input_tensor)

            acc, acc_top3 = self.accuracy(output.data, target, (1,3))

            acc_avg += acc * batch_size
            acc_top3_avg += acc_top3 * batch_size
            total += batch_size
        acc_avg /= total
        acc_top3_avg /= total
        self.writer.add_scalar('Test/Acc', acc_avg, epoch)
        self.writer.add_scalar('Test/Acc_Top3', acc_top3_avg, epoch)

        torch.cuda.empty_cache()

        summary['acc'] = acc_avg
        summary['acc_top3'] = acc_top3_avg

        return summary



    def recognize(self,input:tensor):
        model = self.model
        model.eval()
        if self.video:
            pred = model(input)
        else:
            pred,_ = model(input)
        pred = pred.topk(1,dim=1)
        return int(pred[1])

    def batch_recognize(self,input_loader:dataloader):
        model = self.model
        model.eval()
        results = dict()
        for i, (input_tensor,filenames)in enumerate(input_loader):
            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
            batch_size = input_tensor.size()[0]
            if self.video:
                preds = model(input_tensor)
            else:
                preds,_ = model(input_tensor)
            for filename,pred in zip(filenames,preds):
                results[filename] = int(pred.topk(1,dim=0)[1])
        return results

    def learning_rate(self,epoch,decay):
        self.decay = 0.1 ** ((epoch - 1) // decay)
        learn_rate = self.learn_rate * self.decay
        if learn_rate < 1e-7:
            learn_rate = 1e-7
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size)[0])
        return res