import json
from lumbar.main import start_time
import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

sys.path.append('..')
sys.path.append(".")
from core.disease.data_loader import DisDataLoader
from core.disease.evaluation import Evaluator
from core.disease.model import DiseaseModelBase
from core.key_point import KeyPointModel, NullLoss
from core.structure import construct_studies

from nn_tools import torch_utils

backbone = resnet_fpn_backbone('resnet50', True)
kp_model = KeyPointModel(backbone)
dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))

dis_model.load_state_dict(torch.load("models/baseline.dis_model"))

dis_model.cuda()

# 预测
testA_studies = construct_studies('../../Datasets/lumbar/lumbar_testA50/')

result = []
for study in testA_studies.values():
    result.append(dis_model.eval()(study, True))

start_time = time.time()
with open('predictions/baseline.json', 'w') as file:
    json.dump(result, file)
print('task completed, {} seconds used'.format(time.time() - start_time))
