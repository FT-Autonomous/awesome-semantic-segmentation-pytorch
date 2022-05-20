import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import os
import sys
import argparse
import torch
from torch import nn

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.models.model_zoo import get_segmentation_model

parser = argparse.ArgumentParser(description='Live preview')
parser.add_argument('--save-folder', default='~/.torch/models')
parser.add_argument('--model', default='icnet')
parser.add_argument('--dataset', default='ughent')
parser.add_argument('--backbone', default='resnet50')
parser.add_argument('--input')
args = parser.parse_args()


def mask_to_color(prediction):
    # This really could be an array
    colors = np.array([
        [0, 0, 0], # Background
        [0, 0, 255], # Blue
        [100, 100, 20], # Yellow
        [100, 50, 0], # Large Orange
        [120, 30, 0]  # Orange 
    ], dtype=np.uint8)
    
    width = prediction.shape[0]
    height = prediction.shape[1]
    return colors[prediction.cpu().numpy()]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_segmentation_model(backbone=args.backbone, pretrained_base=False, root=args.save_folder, model=args.model, dataset=args.dataset, norm_layer=nn.BatchNorm2d).to(device)
model.load_state_dict(torch.load('/home/vandewoe@ad.mee.tcd.ie/.torch/models/icnet_resnet50_ughent.pth', map_location=device))

model.eval()
image = cv.cvtColor(cv.imread(args.input), cv.COLOR_BGR2RGB)
normal = torchvision.transforms.Normalize([.485, .456, .406], [.229, .224, .225])
tensor = normal(torch.from_numpy(image.transpose(2, 0, 1)).to(device).float())
print('fully setup now')
pred = model(tensor[None,...])
print('predicted')
mask = mask_to_color(pred[0][0].argmax(0))

image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
cv.imshow("Mask and Image", np.uint8((image + mask) / 2))
cv.imshow("Mask only", np.uint8(cv.cvtColor(mask_to_color(pred[2][0].argmax(0)), cv.COLOR_RGB2BGR) * 20))
cv.waitKey(10000)
