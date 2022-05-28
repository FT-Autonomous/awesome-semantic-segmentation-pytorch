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

from fta.models.model_zoo import get_segmentation_model

parser = argparse.ArgumentParser(description='Live preview')
parser.add_argument('--model', default='cgnet')
parser.add_argument('--dataset', default='ughent')
parser.add_argument('--backbone', default='resnet50')
parser.add_argument('--weights', required=True)
parser.add_argument('--input',
                    help='the path to the input image')
parser.add_argument('--output',
                    help='the path where the mask will be saved')
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

def predict(image, resize=True):
    '''Takes a numpy HWC, BGR image, returns a numpy HWC, BGR image with the user friendly masks'''
    if resize:
        new_height = int(512 * image.shape[1] / image.shape[0])
        image = cv.resize(image, (new_height - new_height % 64, 512), interpolation=cv.INTER_NEAREST)
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    normal = F.Normalize([.485, .456, .406], [.229, .224, .225])
    tensor = normal(torch.from_numpy(rgb_image.transpose(2, 0, 1)).to(device).float())
    pred = torch.nn.Softmax(0)(model(tensor[None, ...])[0][0])
    #pred[0, pred[0, ...] > 0.2] = 1
    return image, cv.cvtColor(mask_to_color(pred.argmax(0)), cv.COLOR_RGB2BGR)

def merge(image, mask, image_vs_mask=0.5):
    image_and_mask = np.uint8(image * image_vs_mask + mask * (1 - image_vs_mask))
    return np.concatenate([image, image_and_mask], 1 if image_and_mask.shape[1] / image_and_mask.shape[0] <= 1.75 else 0)

def show(image, interval=10000):
    cv.imshow("Mask and Image", image)
    cv.moveWindow("Mask and Image", 30, 30)
    #cv.imshow("Mask only", mask)
    #cv.imwrite(f"test/{random.randint(0, 1000000)}.png", result)
    cv.waitKey(interval)

if args.input is None:
    cam = cv.VideoCapture(args.camera)
    while True:
        result, image = cam.read()
        show(merge(*predict(image)))
else:
    original_image = cv.imread(args.input)
    mask = merge(*predict(original_image))
    if args.output:
        cv.imwrite(args.output, mask)
    else:
        show(mask)
