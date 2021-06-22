# -*- coding:utf-8 -*-
import torch
import sys,os
import numpy as np
import model
d = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(d,'premodel')
characters = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
models = model.IMGmodel().cpu()
models.load_state_dict(torch.load(os.path.join(model_dir,"model_params.pkl"), map_location='cpu'))
models.eval()

def convert_to_char(x):
    last_char = None
    res = []
    shengyu = []
    for char, score in x:
        if char != last_char or last_char is None:
            res.append((char, score))
        elif res[-1][1] < score:
            res[-1] = (char, score)
        else:
            shengyu.append((char, score))
        last_char = char
    res.sort(key = lambda x:x[1], reverse = True)
    shengyu.sort(key = lambda x:x[1], reverse = True)
    if len(res) >= 5:
        res = res[:5]
        res.sort(key = lambda i:x.index(i))
        return "".join([characters[i] for i,s in res])
    else:
        res.extend(shengyu[:5 - len(res)])
        res.sort(key = lambda i:x.index(i))
        return "".join([characters[i] for i,s in res])
    
def ocr_predict(im):
    im = im.point(lambda x: 1 if x == 255 else 0, '1')
    datas = []
    for i in range(5,71):
        start = i
        ims = im.crop((start,0,start+14,25))
        datas.append(list(ims.getdata()))
   
    x = torch.tensor(datas, dtype = torch.float32).reshape(len(datas),14,25)
    output = models(x)
    output = torch.nn.functional.softmax(output,dim = 1).tolist()
    output = list(filter(lambda x:x[0] != 0 and x[1] > 0.5,map(lambda x:sorted(enumerate(x),key = lambda x:x[1], reverse = True)[0], output)))
    #return list(map(lambda x:(characters[x[0]],x[1]),output))
    return convert_to_char(output)

if __name__ == "__main__":
    from PIL import Image
    
    count = 0
    _,_,images =  list(os.walk("jiance"))[0]
    for img in images:
        if img[-3:] != 'png':continue
        label = img[:-4].split("_")[1]
        im = Image.open(os.path.join("jiance",img))
        print(ocr_predict(im),label)
        if ocr_predict(im) == label:
            count += 1
        #break
    print(count / len(images))

    #img = Image.open("1565863214194_M9WCT.png")
    #print(ocr_predict(img))
