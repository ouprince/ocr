# -*- coding:utf-8 -*-
from dataset import IMGdata
import torch
from torch.utils import data
import torch.nn.functional as F
import sys,os
import model
import numpy as np
d = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(d,'premodel')
import logging
logger_console = logging.getLogger("multilabels_console")
formatter = logging.Formatter('%(process)s %(levelname)s:  %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger_console.addHandler(console_handler)
logger_console.setLevel(logging.INFO)
characters = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def val(models,dataset):
    models.eval()
    dataset.train(False)
    num_counts = len(dataset)
    dataloader = data.DataLoader(dataset=dataset,batch_size = 1024,shuffle=False,num_workers=4,pin_memory=True)
    
    count = 0.0
    losses = 0.0
    for i,(datas, target) in enumerate(dataloader):
        batch = datas.size(0)
        datas = datas.reshape(batch,14,25)
        datas, target = datas.cuda(), target.cuda()
        output = models(datas)
        loss = F.cross_entropy(output, target)
        losses += loss.item()
        corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        count += corrects
        #print(output_argmax)
        #print(target)
    dataset.train(True)
    models.train()
    return float(count)/num_counts, losses/ (i + 1)

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
        
def ocr_predict(models,im):
    im = im.point(lambda x: 1 if x == 255 else 0, '1')
    datas = []
    for i in range(5,71):
        start = i
        ims = im.crop((start,0,start+14,25))
        datas.append(list(ims.getdata()))
   
    x = torch.tensor(datas, dtype = torch.float32).reshape(len(datas),14,25).cuda()
    output = models(x)
    output = torch.nn.functional.softmax(output,dim = 1).tolist()
    output = list(filter(lambda x:x[0] != 0 and x[1] > 0.5,map(lambda x:sorted(enumerate(x),key = lambda x:x[1], reverse = True)[0], output)))
    #return list(map(lambda x:(characters[x[0]],x[1]),output))
    return convert_to_char(output)
            
from PIL import Image
def val_png(models):
    models.eval()
    count = 0
    _,_,images =  list(os.walk("jiance"))[0]
    for img in images:
        if img[-3:] != 'png':continue
        label = img[:-4].split("_")[1]
        print(label)
        im = Image.open(os.path.join("jiance",img))
        if ocr_predict(models,im) == label:
            count += 1
        #break
    models.train()
    return float(count / len(images)), 1.0
    
        
def train(**kwargs):
    models = model.IMGmodel().cuda()
    models.train() # 训练模型
    dataset = IMGdata(True)
    dataset.train(True)
    print("len(dataset) =",len(dataset))
    dataloader = data.DataLoader(dataset=dataset,batch_size = 1024 * 4,shuffle=True,num_workers=4,pin_memory=True)
    lr = 2e-3
    optimizer = models.get_optimizer(lr = lr)
    if os.path.exists(os.path.join(model_dir,"score.pkl")):
        best_score, best_loss = torch.load(os.path.join(model_dir,"score.pkl"))
    else:
        best_score, best_loss = 0.0, 9999

    if os.path.exists(os.path.join(model_dir,"model_params.pkl")):
        models.load_state_dict(torch.load(os.path.join(model_dir,"model_params.pkl")))
    
    epos = -1
    best_epos = 0
    while True:
        epos += 1
        for i,(datas, target) in enumerate(dataloader):
            batch = datas.size(0)
            datas = datas.reshape(batch,14,25)
            datas, target = datas.cuda(), target.cuda()
            optimizer.zero_grad()
            output = models(datas)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        score,losses = val_png(models)#,dataset)
        if (score >= best_score and losses < best_loss) or score > best_score:
            best_epos, best_score, best_loss = epos, score, losses
            logger_console.info("Saving model,scores = %.4f and loss = %.4f" %(score,losses))
            torch.save(models.state_dict(),os.path.join(model_dir,"model_params.pkl"))
            torch.save((score,losses), os.path.join(model_dir,"score.pkl"))
        elif epos - best_epos > 15:
            best_epos = epos
            if os.path.exists(os.path.join(model_dir,"score.pkl")):
                best_score, best_loss = torch.load(os.path.join(model_dir,"score.pkl"))
            if os.path.exists(os.path.join(model_dir,"model_params.pkl")):
                models.load_state_dict(torch.load(os.path.join(model_dir,"model_params.pkl")))
            lr = lr * 0.95
            optimizer = models.get_optimizer(lr = lr) 

        if epos % 1 == 0:
            logger_console.info("After %d epos, scores = %.4f and loss = %.4f, trainning loss = %.4f" %(epos,score,losses,loss.item())) 
        if score > 0.95 and losses < 0.05:
            torch.save(models.state_dict(),"model_params.pkl")
            break


if __name__ == "__main__":
    train()
