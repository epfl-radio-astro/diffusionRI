import math, torch
import torch.nn as nn
from random import randint

from torchvision import transforms
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import InterpolationMode


import random
import numpy as np




###################

def return_x(X):
    return X


########################

import torchvision.transforms.functional as TF

class ShiftTransform:
    def __init__(self, max_px=20, p=1.0,fill=0):
        self.max_px = max_px
        self.fill = fill
        self.p = p
    def __call__(self, x):
        if torch.rand(()) < self.p:
            dx = torch.randint(-self.max_px, self.max_px+1, (1,)).item()
            dy = torch.randint(-self.max_px, self.max_px+1, (1,)).item()

            x =  TF.affine(
                x,
                angle=0,
                translate=(dx, dy),
                scale=1.0,
                shear=0,
                fill=self.fill
            )
        return x
    

train_tf = transforms.Compose([
    transforms.ToTensor(),
    ShiftTransform(max_px=20, p=1.0, fill=0),
    transforms.RandomRotation(
        degrees=(0, 360),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.8,1.2)),
    transforms.Lambda(lambda x: x.clamp(0,1)),
    transforms.Lambda(lambda x: x*2 - 1.0),
])


DATA_MEAN = 0.5
DATA_STD = 2.0

def rgz_transform_real(arr):
    arr = train_tf(arr)
    return arr

def first_radio_galaxy_transform_real(arr, jitter=False):
    if jitter:
        i, j = randint(-10, 10), randint(-10, 10)
    else:
        i, j = 0, 0

    arr = arr[75+i:75+150+i, 75+j:75+150+j]
    arr = train_tf(arr)
    return arr



def rgz_transform_val(arr):
    return ((arr / 255.0) - DATA_MEAN) * DATA_STD

def first_radio_galaxy_transform_val(arr, jitter=False):
    if jitter:
        i, j = randint(-10, 10), randint(-10, 10)
    else:
        i, j = 0, 0

    arr = arr[75+i:75+150+i, 75+j:75+150+j]
    return ((arr / 255.0) - DATA_MEAN) * DATA_STD 



def get_image_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform



def proj_hermitian(z: torch.Tensor) -> torch.Tensor:
    z_flip = torch.flip(z, dims=(-2, -1))

    z_partner = torch.roll(z_flip, shifts=(1, 1), dims=(-2, -1))

    z_partner = torch.conj(z_partner)

    z_sym = 0.5 * (z + z_partner)

    return z_sym

def save_tall_image_grid(tensor_batch, save_path, images_per_row=5, padding=2):
    """
    Saves a tall grid of grayscale images from a (B, 1, H, W) tensor batch.
    Applies per-image min-max normalization before saving.

    Args:
        tensor_batch (torch.Tensor): Tensor of shape (B, 1, H, W)
        save_path (str): Output file path (e.g., 'output.png')
        images_per_row (int): How many images in each row
        padding (int): Padding between images
    """
    tensor_batch = tensor_batch.detach().cpu()

    # Per-image min-max normalization
    B = tensor_batch.size(0)
    normalized_batch = []
    for i in range(B):
        img = tensor_batch[i]
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = torch.zeros_like(img)
        normalized_batch.append(img)
    tensor_batch = torch.stack(normalized_batch)

    # Create a grid
    grid = make_grid(tensor_batch, nrow=images_per_row, padding=padding)

    # Convert to PIL and save
    image = to_pil_image(grid)
    image.save(save_path)
