# -*- coding:utf-8 -*-
import torch
import sys,os
import numpy as np
import model
d = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(d,'premodel')
characters = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
models = model.IMGmodel().cpu()
models.load_state_dict(torch.load(os.path.join(model_dir,"model_params.pkl")))
models.eval()

import json
from PIL import Image
from flask import Flask,request
from flask import Response
from flask_cors import CORS

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
    return convert_to_char(output)

app = Flask(__name__)
CORS(app, resources=r'/*')
@app.route('/yanzhengma', methods=['POST'])
def hello_world():
    im = request.files['img']
    im = Image.open(im)
    yanzhengma = ocr_predict(im)
    resp = Response(yanzhengma)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    app.run(port=10730,host='0.0.0.0',debug=False)
