# -*- coding:utf-8 -*-
import sys,os

width = 22
height = 34
curdir = os.path.dirname(os.path.realpath(__file__))
dir_name = os.path.join(curdir,"train")

deal_error = False # 这里必须改
datas = []
_,_,images =  list(os.walk(dir_name))[0]
for idx,x in enumerate(images):
    label, flag = x.split(".")[:2]
    if flag != 'png':break
    if deal_error:
        label = label.split("_")[1]
    assert len(label) == 5,"should be 5 chars"
    for i,c in enumerate(label):
        x = 10 + i*width
        y = 11
        x_1 = x + width
        y_1 = y + height
        datas.append([c,str(x),str(y),str(x_1),str(y_1),str(idx)])


	
with open(os.path.join(dir_name,"oyp.font.exp0.box"),"w") as f:
    for d in datas:
        f.write(" ".join(d) + "\n")
