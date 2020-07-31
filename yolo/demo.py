from darknet import *
from utils import *
import torch
import cv2
import torchvision.transforms as transformers
import glob
from PIL import Image

model = Darknet("yolov3.cfg")
model.load_weights("yolov3.weights")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

normalize = transformers.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
trans = transformers.Compose([
    transformers.Resize((608, 608)),
    transformers.ToTensor(),
    normalize
])


for imgPath in glob.glob("example/build/imgs/*.jpg"):

    img_ori = cv2.imread(imgPath)
    img = prep_image(img_ori,(608, 608))

    img = img.to(device)
    pred, probs = model(img)
    
    
    pred = torch.cat((pred,probs),dim=2)

    pred = write_results(pred, 0.5, 80)
    if pred is None:
        continue
    
    if len(pred.shape) == 3:
        pred = torch.squeeze(pred).data.cpu().numpy()
    else:
        pred = pred.data.cpu().numpy()

    scale = min(608 / img_ori.shape[1], 608 / img_ori.shape[0])

    pred[:,[1, 3]] -= (608 - scale * img_ori.shape[1]) / 2
    pred[:, [2, 4]] -= (608 - scale * img_ori.shape[0]) / 2


    for bbox in pred:
        if bbox[-1] == 0:
            pt1 = (int(bbox[1] / scale), int(bbox[2] / scale))
            pt2 = (int(bbox[3] / scale), int(bbox[4] / scale))

            cv2.rectangle(img_ori,pt1,pt2,(0,255,0),2)

    cv2.imshow("detection",img_ori)
    if cv2.waitKey() == 27:
        break