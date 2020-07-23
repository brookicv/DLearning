from darknet import *
from utils import *
import torch
import cv2

model = Darknet("yolov3.cfg")
model.load_weights("yolov3.weights")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
model = model.to(device)


dummy_input = torch.randn(1, 3, 608, 608, device="cpu")

input_names = ["input1"]
output_names = ["output1"]

torch.onnx.export(
    model,
    dummy_input,
    "models/yolov3_608.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names)
