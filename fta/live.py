import cv2 as cv
import numpy as np
import torchvision.transforms as F
import os
import sys
import torch

from .models.model_zoo import get_segmentation_model

def mask_to_color(prediction):
    colors = np.array([
        [0, 0, 0], # Background
        [0, 0, 255], # Blue
        [100, 100, 20], # Yellow
        [100, 50, 0], # Large Orange
        [120, 30, 0]  # Orange 
    ], dtype=np.uint8)

    return colors[prediction]

def scale(image, output_width):
    new_height = int(output_width * image.shape[1] / image.shape[0])
    return cv.resize(image, (new_height - new_height % 64, output_width), interpolation=cv.INTER_NEAREST)

def half(image, output_shape=None):
    halfed_image = image[:, :image.shape[1]//2, :]
    return cv.resize(halfed_image, output_shape) if output_shape else halfed_image

def predict_raw(image, model, device):
    '''Takes an OpenCV BGR image and returns class probabilities with the shape HxWx5'''
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    normal = F.Normalize([.485, .456, .406], [.229, .224, .225])
    tensor = normal(torch.from_numpy(rgb_image.transpose(2, 0, 1)).to(device).float())
    return torch.nn.Softmax(0)(model(tensor[None, ...])[0][0])

def predict(*args, **kwargs):
    return predict_raw(*args, **kwargs).argmax(0).cpu().numpy()

def colorise(pred):
    '''Takes predictions returned from `predict` and returns a BGR image with the shape HxWx3'''
    return cv.cvtColor(mask_to_color(pred), cv.COLOR_RGB2BGR)

def merge(image, mask, image_vs_mask=0.5):
    image_and_mask = np.uint8(image * image_vs_mask + mask * (1 - image_vs_mask))
    return np.concatenate([image, image_and_mask], 1 if image_and_mask.shape[1] / image_and_mask.shape[0] <= 1.75 else 0)

def show(image, interval=10000):
    cv.imshow("Mask and Image", image)
    cv.moveWindow("Mask and Image", 30, 30)
    cv.waitKey(interval)

def do_it_all(model, device, cam, image_vs_mask=0.5):
    result, image = cam.read()
    image = half(image, (1792//2, 512))
    show(merge(image, colorise(predict(image, model, device)), image_vs_mask=image_vs_mask), interval=1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Live preview')
    parser.add_argument('--model', default='cgnet',
                        help='The name of the model to load OR the path to a torchscript model containing both the computational graph and the weights')
    parser.add_argument('--dataset', default='ughent') # The dataset just gives the number of classes...
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--weights')
    parser.add_argument('--input',
                        help='the path to the input image')
    parser.add_argument('--output',
                        help='the path where the mask will be saved')
    parser.add_argument('--camera', type=int, default=1,
                        help='which camera to use')
    parser.add_argument('--debug', action='store_true',
                        help='Drops into the debug prompt after the model has been loaded')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image-intensity', type=float, default=0.5,
                        help='multiplier that will be applied to the image when it\'s added to the mask')
    group.add_argument('--mask-intensity', type=float, default=0.5,
                        help='multiplier that will be applied to the mask when it\'s added to the image')
    args = parser.parse_args()

    if args.mask_intensity != 0.5:
        intensity = 1 - args.mask_intensity
    else:
        intensity = args.image_intensity
    assert 0 <= intensity and intensity <= 1, "Intensity must be in [0, 1]"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model.endswith(".ts"):
        model = torch.jit.load(args.model, map_location=device)
    else:
        model = get_segmentation_model(backbone=args.backbone, pretrained_base=False, model=args.model, dataset=args.dataset, norm_layer=torch.nn.BatchNorm2d).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model = torch.jit.script(model)
    model.train()

    if args.debug:
        breakpoint()
        
    if args.input is None:
        cam = cv.VideoCapture(args.camera)
        while True:
            do_it_all(model, device, cam, intensity)
    else:
        original_image = cv.imread(args.input)
        mask = merge(original_image, colorise(predict(original_image, model, device)))
        if args.output:
            cv.imwrite(args.output, mask)
        else:
            show(mask)
