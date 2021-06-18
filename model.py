# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class IMGmodel(nn.Module):
    def __init__(self, input_shape=(3, 64, 128)):
        super(IMGmodel, self).__init__()
        input_shape = input_shape
        self.filter_sizes = [3,5,7]
        self.convs1 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(1, 32, size),nn.BatchNorm2d(32)) for size in self.filter_sizes])

        self.convs2 = nn.ModuleList(
                    [nn.Sequential(nn.Conv2d(32, 64, size, padding = int((size-1)/2)),nn.BatchNorm2d(64)) for size in self.filter_sizes])

        self.convs3 = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(64, 128, size, padding = int((size - 1)/2)),nn.BatchNorm2d(128)) for size in self.filter_sizes])

        self.downsamples = nn.ModuleList(
                                  [nn.Sequential(nn.Conv2d(1, 128, size),nn.BatchNorm2d(128)) for size in self.filter_sizes])

        self.fc = nn.Sequential(nn.Linear(384, 200), nn.ReLU(True), nn.Linear(200,100), nn.ReLU(True), nn.Linear(100,10))

    def get_optimizer(self,lr = 5e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 0)
        return optimizer

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一维，变成 [batchsize,1,seq,embedding]
        downsamples = [conv(x) for conv in self.downsamples]
        x = [nn.ReLU(inplace = True)(conv(x)) for conv in self.convs1]
        x = [nn.ReLU(inplace = True)(conv(i)) for conv,i in zip(self.convs2,x)]
        x = [nn.ReLU(inplace = True)(conv(i) + downsample) for conv,downsample,i in zip(self.convs3,downsamples,x)]
        x = [i.reshape(i.size(0),i.size(1),-1) for i in x]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x,1)
        x = self.fc(x)
        return x
