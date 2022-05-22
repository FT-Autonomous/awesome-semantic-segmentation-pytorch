from PIL import Image
import numpy as np
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from skimage.util import random_noise

# Some basic augmentations
def augment(img, mask, p_affine=0.5, p_gauss=0.7, p_salt_and_pepper=0.5, p_random_erase=0.5, p_speckle=0.7):        
    # Gaussian Blur
    if random.random() < p_gauss:
        img = T.GaussianBlur(kernel_size=random.randrange(9,25,2), sigma=(2.0, 2.5))(img)
        
    # Random Affine
    if random.random() < p_affine:
        affine_params = T.RandomAffine.get_params((-8,8), (0.0, 0.0), (1.0, 1.0), (-8, 8), img.size)
        img = F.affine(img, *affine_params, interpolation=T.InterpolationMode.NEAREST)
        mask = F.affine(mask, *affine_params, interpolation=T.InterpolationMode.NEAREST)
            
    # Salt and Pepper
    if random.random() < p_salt_and_pepper:
        img = np.array(img)
        blank = np.zeros(img.shape[:2])
        pepper_map = random_noise(blank, mode='s&p', salt_vs_pepper=1, amount=0.025)
        salt_map = random_noise(blank, mode='s&p', salt_vs_pepper=1, amount=0.025)
        img[pepper_map == 1] = 0
        img[salt_map == 1] = 255
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

    # Speckle image
    if random.random() < p_speckle:
        img = np.asarray(img)
        img = random_noise(img, mode='speckle', mean=0.2, var=0.2)
        img = Image.fromarray(np.uint8(255 * img))
        
    return img, mask
