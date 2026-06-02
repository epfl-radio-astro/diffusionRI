# import statements & checking for cuda
import torch
torch.set_float32_matmul_precision('high')
from real_valued.trainer import LitDDPM
from real_valued.unet import Unet
from helpers import first_radio_galaxy_transform_val, proj_hermitian
from torch.utils.data import DataLoader
from dataset.radio_galaxy_dataset.firstgalaxydata import FIRSTGalaxyData

import numpy as np
from ddrm.svd_replacement import Fourier2D
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from matplotlib.colors import LogNorm

device = torch.device('cuda')

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print("Current device:", idx)
    print("Name:", torch.cuda.get_device_name(idx))
    x = torch.randn(1, device=f"cuda:{idx}")
    print("Tensor device:", x.device)

import os

# defining the model

# the image size we use will be 150x150, as set by the pre-trained DDPM
image_size = 150

# define and load the pre-trained DDPM
net = Unet(1, 32, 4, 2, 256).to('cuda')
model = LitDDPM.load_from_checkpoint('/capstor/store/cscs/ska/sk031/last-v5.ckpt', unet=net)

# define the sampling operator used by DDRM

# sampling operator H
mask = torch.from_numpy(np.load('uv_coverages/VLA_mask_150.npy'))
h = Fourier2D(channels=1, img_dim=image_size, S=mask, device=device)

# number of DDRM sampling steps
# increase this number to improve the image reconstruction
num_steps = [1000]#[10,100,1000]

# for a single input image, we will denoise batch_size times
# increase this number to get more samples
batch_size = 1

# sigma is the amount of noise added to visibility space
#sigma_0s = [0.0,0.01,0.1,0.2,0.5]
sigma_0s = [0.0,0.01,0.02,0.03, 0.04, 0.05,0.1]
sigma_0s = [0.0,0.01, 0.05,0.1]

# the eta hyperparamt 
eta_A, eta_B, eta_C = 0.85, 1.0, 0.85

'''
# for our test data we choose an out-of-domain image
# however, we need to trim/resize this down to our 150x150 input image size
test_image_file = "/capstor/scratch/cscs/etolley/3c353_gdth.fits"

from astropy.io import fits

with fits.open(test_image_file) as hdul:
    # this image is 512 x 512, rescale to 150x150
    true_image2 = hdul[0].data[::3,::3].astype(np.float32)
    true_image2 = true_image2[10:-11,10:-11]
    true_image1 = hdul[0].data[200:350,10:160].astype(np.float32)
assert true_image1.shape == (150, 150) and true_image2.shape == (150, 150)

outdir = "/capstor/scratch/cscs/etolley/image_3c353/"
Path(outdir).mkdir(parents=True, exist_ok=True)
'''
outdir = "/capstor/scratch/cscs/etolley/image_noiseradiogalaxies/"
Path(outdir).mkdir(parents=True, exist_ok=True)

test_data = FIRSTGalaxyData(root="/capstor/scratch/cscs/etolley/ddrm_data/", selected_split="test", selected_classes=["FRI", "FRII", "Compact", "Bent"], input_data_list=["galaxy_data_h5.h5"],
                           is_PIL=False, is_RGB=True, transform=first_radio_galaxy_transform_val)

test_dl = DataLoader(
    test_data, 
    batch_size=1, 
    shuffle=False, 
    drop_last=False, 
)


for steps in num_steps:
    for sigma_0 in sigma_0s:
        #for i,true_image in enumerate([true_image1,true_image2]):
        for i,  x in enumerate(test_dl):

    
            # #rescale the input image to go from 0 to 1
            # shift = np.min(true_image)
            # scale = np.max(true_image)
            # true_image = (true_image-shift)/(scale-shift)
    
            # np.save(outdir+"true"+file_postfix, true_image)
    
            # # format the true image x_0
            # x = torch.from_numpy(true_image).reshape(1,image_size, image_size)
            np.save(outdir+"true_{0}".format(i), x.numpy())
            x_gpu = x.to(device)
            x_0 = x_gpu.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # define the observation y_0
            y_0 = h.H(x_0)
            y_0 = y_0 + sigma_0 * proj_hermitian(torch.randn_like(y_0) * np.sqrt(2))
            
            # do the DDRM sampling knowing the noise level
            with model.ema.average_parameters():
                model.model.eval()
                x_hat, _ = model.model.sample_ddrm_y0(
                    image_size, steps, y_0, x_0.shape[0], h, sigma_0, eta_A, eta_B, eta_C
                )
            file_postfix = "_steps{2}_sigma{0:0.2f}_{1}.npy".format(sigma_0,i,steps)
            # rescale the image estimate x_hat
            x_hat = ((x_hat[-1] + 1) / 2).clamp(0, 1) 
            np.save(outdir+"ddrm_sigmacorr"+file_postfix, x_hat.numpy())
    
            # do the DDRM sampling without knowing the noise level
            with model.ema.average_parameters():
                model.model.eval()
                x_hat_0, _ = model.model.sample_ddrm_y0(
                    image_size, steps, y_0, x_0.shape[0], h, 0.0, eta_A, eta_B, eta_C
                )
            
            # rescale the image estimate x_hat
            x_hat_0 = ((x_hat_0[-1] + 1) / 2).clamp(0, 1) 
            np.save(outdir+"ddrm_sigma0"+file_postfix, x_hat_0.numpy())

            file_postfix = "_sigma{0:0.2f}_{1}.npy".format(sigma_0,i)
            x_dirty = h.H_pinv(y_0).reshape(batch_size,image_size,image_size).cpu()
            np.save(outdir+"dirty"+file_postfix, x_dirty)