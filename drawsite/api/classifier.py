import os 
import base64
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = None
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path,"./models/mnist_cnn.pt")

model_input_size = 28, 28

def load_model():
    global model
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()

def get_data():
    if model is None:
        raise Exception("Expensive model not loaded")
    return model

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def classify_img(b64_img):
    if "data:image" in b64_img:
        b64_img = b64_img.split(",")[1]
    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(b64_img)
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)
    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    #image = mpimg.imread(image_stream)

    image.thumbnail(model_input_size, Image.Resampling.LANCZOS)
    image = image.convert('P')
    image = np.array([[image]], dtype=np.float32)
    image[image != 0] = 1
    
    img_input = torch.tensor(image).to(device)
    output = model(img_input)
    print("raw probabilities", output)
    img_class = output.argmax(dim=1, keepdim=True)
    print("selected class", img_class)

    return str(img_class.detach().cpu().numpy()[0][0])