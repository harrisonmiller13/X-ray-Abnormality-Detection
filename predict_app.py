import matplotlib as plt
import numpy as np
import io
import json

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask, render_template



def get_model():
    path = torch.load('best_BCE.dense.tar', 
                        map_location=torch.device('cpu'))
    model = models.densenet201(pretrained=True)
    model.classifier = nn.Linear(1920,2)
    model.load_state_dict(path['state_dict'],strict=False)
    model.eval()
    print(' 🔥Loaded PyTorch model 🔥ヘ(◕。◕ヘ) ')
    return model
    
    

def preprocess_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485,0.456,0.406],
                                            [0.229,0.224,0.225]
                                        )])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

with open('class_to_idx.json') as f:
    class_to_result = json.load(f)

idx_to_class = {v: k for k,v in class_to_result.items()}

model = get_model()

def get_inference(image_bytes):
    tensor = preprocess_image(image_bytes)
    outputs = model.forward(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    class_idx = idx_to_class[category]
    return category,class_idx



app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', value='hello')
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not uploaded")
            return
        file = request.files['file']
        image = file.read()
        category, class_idx = get_inference(image_bytes=image)
        return render_template('result.html', xrayresult= class_idx, result=category)

if __name__=='__main__':
    app.run(debug=True)