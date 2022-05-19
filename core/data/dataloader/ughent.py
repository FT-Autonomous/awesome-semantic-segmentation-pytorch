"""Pascal VOC Semantic Segmentation Dataset."""

import matplotlib.pyplot as plt
import matplotlib

import argparse
import os
import torch
import numpy as np
import random

from PIL import Image
from .segbase import SegmentationDataset
from glob import glob

class UGhentSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to supervisely data folder. Default is './datasets/ughent/images/'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = UGhentSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = 5

    def __init__(self, root='../datasets/ughent', split='train', mode=None, transform=None, **kwargs):
        super().__init__(root, split, mode, transform, **kwargs)
        self.num_samples = 10
        self.images = sorted(glob(os.path.join(root, "img/*")))
        self.masks = sorted(glob(os.path.join(root, "masks_machine/*")))
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        
        # synchronized transform
        if self.mode == 'train':
            img, mask = self.augment(img, mask)
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('background', 'blue_cone', 'yellow_cone', 'large_orange_cone', 'orange_cone')

    # Some basic augmentations
    def augment(self, img, mask):
        import torchvision.transforms as T
        import torchvision.transforms.functional as F
        from skimage.util import random_noise
        
        p_gauss = 0.5
        p_affine = 0.5
        p_random_crop = 0.5
        p_salt_and_pepper = 0.5
        p_random_erase = 0.5
        
        # Gaussian Blur
        if random.random() < p_gauss:
            img = T.GaussianBlur(kernel_size=random.randrange(5,25,2), sigma=(0.3, 3.0))(img)
            
        # Random Affine
        if random.random() < p_affine:
            affine_params = T.RandomAffine.get_params((-8,8), (0.05,0.05), (0.95, 0.95), (-8, 8), img.size)
            img = F.affine(img, *affine_params, interpolation=T.InterpolationMode.NEAREST)
            mask = F.affine(mask, *affine_params, interpolation=T.InterpolationMode.NEAREST)
            
        # Random crop
        if random.random() < p_random_crop:
            source_width, source_height = (random.randrange(img.size[0], int(img.size[0]*1.5)),
                                           random.randrange(img.size[1], int(img.size[1]*1.5)))
            resize = T.Resize(size=(source_height, source_width))
            crop_params = T.RandomCrop.get_params(img, output_size=(img.size[1], img.size[0]))
            img = F.crop(resize(img), *crop_params)
            mask = F.crop(resize(mask), *crop_params)
            
        # Salt and Pepper
        if random.random() < p_salt_and_pepper:
            img = np.array(img)
            blank = np.zeros(img.shape[:2])
            pepper_map = random_noise(blank, mode='s&p', salt_vs_pepper=1, amount=0.05)
            salt_map = random_noise(blank, mode='s&p', salt_vs_pepper=1, amount=0.05)
            img[pepper_map == 1] = 0
            img[salt_map == 1] = 1
            img = Image.fromarray(img)
            
        # Random erase
        if random.random() < p_random_erase:
            img = F.to_tensor(img)
            mask = F.to_tensor(mask).repeat(3,1,1) # For whatever reason, F.erase expects a three dimensional image
            random_erasing_params = T.RandomErasing.get_params(img, ratio=(0.5, 1.5), scale=(0.01, 0.02), value=[0])
            F.erase(img, *random_erasing_params, inplace=True)
            F.erase(mask, *random_erasing_params, inplace=True)
            img = F.to_pil_image(img)
            mask = F.to_pil_image(mask[0, ...])

        return img, mask

def convert_prediction_to_color_mask(prediction):
    # This really could be an array
    colors= [
        [0, 0, 0], # Background
        [0, 0, 255], # Blue
        [100, 100, 20], # Yellow
        [100, 50, 0], # Large Orange
        [120, 30, 0]  # Orange 
    ]
    
    width = prediction.shape[0]
    height = prediction.shape[1]
    color_mask = np.zeros((width, height, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color_mask[x, y] = colors[prediction[x, y]]
    return color_mask


def sample_for_display(dataset, image_to_mask=0.4):
    img, mask, _ = dataset[random.randint(0, len(dataset) - 1)]
    image_part = np.uint8(image_to_mask * img)
    mask_part = np.uint8((1 - image_to_mask) * convert_prediction_to_color_mask(mask))
    return mask_part + image_part

def main():
    matplotlib.use('TkAgg')
    
    parser = argparse.ArgumentParser(description='Test the UGhent data loader')
    parser.add_argument('--path', type=str,
                        help='a path to the "dataset" subfolder of the supervisely output project')
    parser.add_argument('--augment', action='store_true',
                        help='whether or not to apply image augmentations')
    parser.add_argument('--once', dest='number_to_generate', action='store_const', const=1,
                        help='one image will be displayed and the program will exit')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='the program will generate images in this folder rather than displaying them with matplotlib')
    parser.add_argument('--number-to-generate', type=int, default=None,
                        help='the number of images to generate')
    parser.add_argument('--geometry', metavar='ROWSxCOLUMNS', type=str, default='1x1',
                        help='the number of rows and columns that the figures should have')
    parser.add_argument('--debug', action='store_true',
                        help='include a breakpoint right after arguments are parsed')
    args = parser.parse_args()

    if args.debug:
        breakpoint()
    
    dataset = UGhentSegmentation(args.path, split='train' if args.augment else 'val')
    
    rows, columns = [int(dimension) for dimension in args.geometry.split('x')]
    
    index = 0
    
    while args.number_to_generate is None or index < args.number_to_generate:
        figure = plt.figure()
        for i in range(1, 1 + rows * columns):
            axis = figure.add_subplot(rows, columns, i)
            axis.imshow(sample_for_display(dataset))
        if args.output_dir is None:
            plt.show()
        else:
            figure.figsave(os.path.join(output_dir,f"{i}.png"))
        index += 1

if __name__ == '__main__':
    main()
