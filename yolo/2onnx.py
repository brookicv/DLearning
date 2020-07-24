from darknet import *
from utils import *
import torch
import cv2

model = Darknet("yolov3.cfg")
model.load_weights("yolov3.weights")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

dummy_input = torch.randn(1, 3, 608, 608, device="cuda")

preds,props = model(dummy_input)
print(preds.shape)
print(props.shape)


input_names = ["input1"]
output_names = ["bboxes","cls_prob"]

torch.onnx.export(
    model,
    dummy_input,
    "models/yolov3_608.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names)

import onnx
model = onnx.load("models/yolov3_608.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))



