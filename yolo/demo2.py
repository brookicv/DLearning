from darknet53 import *
from utils import *
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transformers
import glob
from PIL import Image

import onnxruntime as rt
import  numpy as np

import onnx

img_ori = cv2.imread("person.jpg")
img = prep_image(img_ori, (608, 608))

np.savetxt("img.txt",img.data.numpy()[0][0],delimiter=",")

sess = rt.InferenceSession("models/sim_yolov3_608.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

pred = sess.run(["bboxes"], {"input1": img.data.numpy()})[0]

print(pred.shape)

np.savetxt("pred2.txt",pred[0],delimiter=",")