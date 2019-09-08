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

app = Flask(__name__)

@app.route('/predict', methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('predict2.html', value='hello')
    if request.method == 'POST':
        predicted_result = 
        return render_template('predict3.html', result=)

if __name__=='__main__':
    app.run(debug=True)
