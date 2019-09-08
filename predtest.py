import matplotlib as plt
import numpy as np
import io

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
    path = 'model_retrieval.tar'
    model = models.densenet201(pretrained=True)
    model.classifier = nn.Linear(1920,1)
    model.load_state_dict(torch.load(
        path, map_location='cpu'),strict=False)
    model.eval()
    return model
    
    print('* Loaded PyTorch model 🔥ヘ(◕。◕ヘ) ')

def preprocess_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                        transforms.Normalize(
                                            [0.485,0.456,0.406],
                                            [0.229,0.224,0.255]
                                        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

model = get_model()

def get_inference(image_bytes):
    tensor = preprocess_image(image_bytes)
    outputs = model.forward(tensor)
    print(outputs)


app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('predict2.html', value='hello')
    if request.method == 'POST':
        if 'file' not in request.files:
            print("file not uploaded")
            return
        file = request.files['file']
        image = file.read()
        get_inference(image_bytes=image)
        return render_template('predict3.html', result='stuff')

if __name__=='__main__':
    app.run(debug=True)