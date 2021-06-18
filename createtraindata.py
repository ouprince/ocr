# -*- coding:utf-8 -*-
import sys,os
from PIL import Image
import numpy as np

curdir = os.path.dirname(os.path.realpath(__file__))
dir_name = os.path.join(curdir,"train")

datas = []
labels = []
characters = '0123456789'
num_classes = 10

def convert2L(x):
    print (x)
    return 1
    #return max([x1,x2,x3,x4])

def process(dir_name,box_name = "oyp.font.exp0.box",limit = None):
    _,_,images =  list(os.walk(dir_name))[0]
    last_idx = None
    with open(os.path.join(dir_name,box_name), encoding = "utf-8") as f:
        for line in f:
            char, x, y, width, height,idx = line.strip().split()
            x,y,width,height,idx = int(x), int(y), int(width), int(height), int(idx)
            if limit is not None and idx > limit:break
            img = Image.open(os.path.join(dir_name,images[idx]))
            for xi in range(x,x+3):
                im = img.crop((xi,y,width+(xi-x),height))
                data = [max(x) for x in im.getdata()]
                data = [1.0 if x >= 150 else 0.0 for x in data]
                datas.append([data,characters.index(char)])

			
process(dir_name,'oyp.font.exp0.box',199)

print(len(datas))

from random import shuffle
for x in range(10):
    shuffle(datas)
	
datas, labels = zip(*datas)		
assert len(datas) == len(labels) ,"should be same length"
print("len(datas) =",len(datas))

'''
for img in images:
    label = img[:-4].split("_")[1]
    im = Image.open(os.path.join("dev",img))
    im = im.point(lambda x: 1 if x == 255 else 0, '1')
    im_data = list(im.getdata())
    assert len(im_data) == 2250 ,"size error"
    if len(label) != 5:
        print(img)
    target = [characters.find(x) for x in label]
    datas.append(im_data)
    labels.append(target)

assert len(datas) == len(labels) , "data = labels"
'''
print("saving datas ... ")
np.savez_compressed("data/train.npz",data = np.array(datas),label = np.array(labels))

