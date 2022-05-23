import cv2 as cv
import numpy as np
from PIL import Image
import torchvision.transforms as F
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
parser.add_argument('--model', default='cgnet')
parser.add_argument('--dataset', default='ughent')
parser.add_argument('--backbone', default='resnet50')
parser.add_argument('--weights', required=True)
parser.add_argument('--input',
                    help='the path to the imput image')
parser.add_argument('--camera', type=int, default=0,
                   help='which camera to use')
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
model = get_segmentation_model(backbone=args.backbone, pretrained_base=False, model=args.model, dataset=args.dataset, norm_layer=nn.BatchNorm2d).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

model.train()

def predict_and_show(original_image, interval=25):
    '''Takes a numpy HWC, BGR image, makes predictions
    And shows them on the screen'''
    new_height = int(512 * original_image.shape[1] / original_image.shape[0])
    resized_image = cv.resize(original_image, (new_height - new_height % 64, 512), interpolation=cv.INTER_NEAREST)
    image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    normal = F.Normalize([.485, .456, .406], [.229, .224, .225])
    tensor = normal(torch.from_numpy(image.transpose(2, 0, 1)).to(device).float())
    pred = torch.nn.Softmax(0)(model(tensor[None, ...])[0][0])
    #pred[0, pred[0, ...] > 0.2] = 1
    mask = cv.cvtColor(mask_to_color(pred.argmax(0)), cv.COLOR_RGB2BGR)
    #cv.imshow("Image only", resized_image)
    result = np.uint8(resized_image // 4 + 3.0 * mask / 2.0)
    cv.imshow("Mask and Image", result)
    #cv.imshow("Mask only", mask)
    #cv.imwrite(f"test/{random.randint(0, 1000000)}.png", result)
    cv.waitKey(interval)

if args.input is None:
    cam = cv.VideoCapture(args.camera)
    while True:
        result, original_image = cam.read()
        predict_and_show(original_image)
else:
    original_image = cv.imread(args.input)
    predict_and_show(original_image, interval=100000)
