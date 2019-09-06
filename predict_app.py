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
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    # device = torch.device('cpu')
    model = models.densenet121(pretrained=True)
    # model.load_state_dict(torch.load(PATHTOMODEL, map_location=device))
    model.eval()
    
    print('* Loaded PyTorch model 🔥ヘ(◕。◕ヘ) ')

def preprocess_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485,0.456,0.406],
                                            [0.229,0.224,0.225]
                                        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = preprocess_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat

get_model()

@app.route("/predict",methods=["GET","POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    # decoded = base64.b64decode(encoded)
    # image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image)

    prediction = model.eval(processed_image).tolist()

    response = {
        'prediction': {
            'normal': prediction[0][0],
            'abnormal': prediction[0][1]
        }
    }
    return jsonify(response)