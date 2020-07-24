from darknet53 import *
from utils import *
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transformers
import glob
from PIL import Image


model = Darknet53(num_classes=80)
state = torch.load("yolov3.pth")


for k,v,m in zip(state.items(), model.modules()):
    if isinstance(m, nn.Conv2d):
        pass

    if isinstance(m, nn.BatchNorm2d):
        pass

for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        pass

    if isinstance(m, nn.BatchNorm2d):
        pass

for k, v in model.named_parameters():
    print(k)

for k, v in state.items():
    print(k)