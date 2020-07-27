from darknet53 import *
from utils import *
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transformers
import glob
from PIL import Image


import onnx
model = onnx.load("models/yolov3_608.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))