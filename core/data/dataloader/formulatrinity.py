"""Pascal VOC Semantic Segmentation Dataset."""


import argparse
import os
import torch
import numpy as np
import random

from PIL import Image
from .segbase import SegmentationDataset
from glob import glob

from .augmentations import augment

class FormulaTrinitySegmentation(SegmentationDataset):
    """Loads fron a directory strucured with two folders: imgs and masks.

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

    def __init__(self, root='datasets/ughent', split='train', mode=None, transform=None, **kwargs):
        super().__init__(root, split, mode, transform, **kwargs)
        self.num_samples = 10
        self.p_affine = 0.5 if 'ughent' not in root else 0
        self.images = sorted(glob(os.path.join(root, "imgs/*")))
        self.masks = sorted(glob(os.path.join(root, "masks/*")))
        
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
            img, mask = augment(img, mask, p_affine=self.p_affine)
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
