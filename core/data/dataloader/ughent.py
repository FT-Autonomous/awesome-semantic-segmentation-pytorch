"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset
from glob import glob

def convert_supervisely_mask(mask):
    # Supervisely exports labels as an RGB image, however, for machine
    # masks, all components are set to the same value.
    supervisely_class_mask = np.asarray(mask)[..., 0].transpose(1, 0)
    sensible_class_mask = np.zeros(mask.size[:2], dtype=float)
  
    mask_map = {
        0 : 0, # Background
        1 : 1, # Blue cone
        3 : 2, # Yellow cone
        9 : 3, # Large orange cone
        10 : 4 # Orange cone
        # 2 : 5, # Unknown cone ?
    }
    
    for y in range(mask.size[0]): # H
        for x in range(mask.size[1]): # W
            pixel_class = mask_map.get(supervisely_class_mask[y][x])
            if pixel_class is not None:
                sensible_class_mask[y][x] = pixel_class

    return Image.fromarray(sensible_class_mask)

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
        self.images = glob(os.path.join(root, "img/*"))
        self.masks = glob(os.path.join(root, "masks_machine/*"))
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = convert_supervisely_mask(Image.open(self.masks[index]))
        # synchronized transform
        if self.mode == 'train':
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

if __name__ == '__main__':
    dataset = UGhentSegmentation()

