#coding:utf-8
import sys,os
from torch.utils import data
import torch as t
import numpy as np
curdir = os.path.dirname(os.path.abspath(__file__))

class IMGdata(data.Dataset):
    def __init__(self, augument=True):
        self.augument = augument
        #self.input_length = input_length
        #self.target_length = target_length
        self.datas = np.load(os.path.join(curdir,'data','train.npz'), allow_pickle=True)['data']
        self.labels = np.load(os.path.join(curdir,'data','train.npz'), allow_pickle=True)['label']
        #self.dev_datas = np.load(os.path.join(curdir,'data','dev.npz'), allow_pickle=True)['data']
        #self.dev_labels = np.load(os.path.join(curdir,'data','dev.npz'), allow_pickle=True)['label'] 
        self.train_data = self.datas, self.labels
        self.val_data = self.datas, self.labels
        
    def dropout(self, d, p = 0.1):
        len_ = len(d)
        index = np.random.choice(len_, int(len_ * p))
        d[index] = 1
        return d

    def train(self, train = True):
        if train:
            self.training = True
            self.datas, self.labels = self.train_data
            self.len_ = len(self.datas)
        else:
            self.training = False
            self.datas, self.labels = self.val_data
            self.len_ = len(self.datas)
    
    def __getitem__(self, idx):
        data = self.datas[idx]
        label = self.labels[idx]
        #if self.training and self.augument:
            #data = self.dropout(data) # 暂不进行数据增强
        #input_length = t.full(size=(1, ), fill_value=self.input_length, dtype=t.long)
        #target_length = t.full(size=(1, ), fill_value=self.target_length, dtype=t.long)
        return t.from_numpy(data).float(), t.tensor(label,dtype = t.long)

    def __len__(self):
        return self.len_


if __name__ == "__main__":
    dataset = IMGdata(True)
    dataset.train(False)
    for data, label in dataset:
        print(data)
        print(label)
