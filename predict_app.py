import matplotlib as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

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
    device = torch.device('cpu')
    model = TheModelClass(*args,**kwargs)
    model.load_state_dict(torch.load(PATH TO MODEL, map_location=device))
    print('* Model Loaded ヘ(◕。◕ヘ) ')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis =0)

    return image

print("* Loading PyTorch model...")

get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size = (224,224))

    prediction = model.eval(processed_image).tolist()

    response = {
        'prediction': {
            'normal': prediction[0][0],
            'abnormal': prediction[0][1]
        }
    }
    return jsonify(response)