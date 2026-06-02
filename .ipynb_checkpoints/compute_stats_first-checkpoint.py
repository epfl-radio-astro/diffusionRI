import torch
torch.set_float32_matmul_precision('high')
from ddpmd.trainer import LitDDPM
from ddpm.unet import Unet
from helpers import first_radio_galaxy_transform_val
from torch.utils.data import DataLoader
from dataset.radio_galaxy_dataset.firstgalaxydata import FIRSTGalaxyData

import numpy as np
from ddrm.svd_replacement import Fourier2D
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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




Path("test_results_b").mkdir(parents=True, exist_ok=True)


batch_size = 128

image_size = 150
model = Unet(1, 32, 4, 2, 256).to('cuda')
t = torch.randint(0, 200, (batch_size,), device=torch.device('cuda'), dtype=torch.long)
# Initialize lazy init
model(torch.randn(batch_size, 1, image_size, image_size, device=torch.device('cuda')),  t)
print('forward pass done')






test_data = FIRSTGalaxyData(root="./data/", selected_split="test", selected_classes=["FRI", "FRII", "Compact", "Bent"], input_data_list=["galaxy_data_h5.h5"],
                           is_PIL=False, is_RGB=True, transform=first_radio_galaxy_transform_val)

#test_data = Subset(test_data, list(range(2)))


test_dl = DataLoader(
    test_data, 
    batch_size=1, 
    shuffle=False, 
    drop_last=False, 
)


def save_samples(save_dir : str, imgs):
    imgs = imgs.permute(0,2,3,1).squeeze(3)
    for i in range(imgs.shape[0]):
        img = imgs[i]
        image = img.detach().cpu().clone()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_axis_off()
        im = ax.imshow(image, cmap='viridis', vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)   
        plt.savefig(save_dir + f"sample_{i}.png" , 
                    bbox_inches='tight', 
                    pad_inches=0.05,     
                    dpi=300)             
        plt.close()


def save_samples_torch(save_dir: str, imgs):
    imgs = imgs.detach().cpu()

    for i in range(imgs.shape[0]):
        sample = imgs[i].clone()
        file_path = os.path.join(save_dir, f"sample_{i}.pt")
        torch.save(sample, file_path)


def save_single_sample(save_dir: str, save_name: str, img, vmin, vmax, norm=None):
    img  = img.squeeze(0)

    image = img 

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_axis_off()
    if norm is not None:
        im = ax.imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(image, cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)   
    plt.savefig(save_dir + save_name+'.png', 
                bbox_inches='tight', 
                pad_inches=0.05,     
                dpi=300)             
    plt.close()

def save_single_sample_torch(save_dir: str, save_name: str, img,):

    img = img.squeeze(0)

    sample = img.detach().cpu().clone()

    # Allow user to pass in names with or without extension
    if not (save_name.endswith(".pt") or save_name.endswith(".pth") or save_name.endswith(".npy")):
        # default to torch format
        filename = save_name + ".pt"
    else:
        filename = save_name

    file_path = os.path.join(save_dir, filename)

    if filename.endswith(".npy"):
        import numpy as np
        np.save(file_path, sample.numpy())
    else:
        torch.save(sample, file_path)


trainer_model = LitDDPM.load_from_checkpoint('checkpoints_real_valued/last-v5.ckpt', unet=model)

## Sampling mask
mask = torch.from_numpy(np.load('uv_coverages/uv_sampling_60_steps.npy'))
model = trainer_model.model
bin_edges = torch.linspace(0.0, 1.0, 21)
n_bins = len(bin_edges) - 1



num_samples_step = [10, 50, 100, 500, 1000]

pixel_stats = {
    num_steps: {
        "se_sum": torch.zeros(n_bins, dtype=torch.float64),
        "count": torch.zeros(n_bins, dtype=torch.long),
    }
    for num_steps in num_samples_step
}

results = []
h = Fourier2D(channels=1, img_dim=image_size, S=mask, device=device)

for i, x in enumerate(test_dl):
    x_gpu = x.to(device)
    x_0 = x_gpu.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    with trainer_model.ema.average_parameters():
        model.eval()

        for num_steps in num_samples_step:

            x_hat, _ = model.sample_ddrm(
                image_size, num_steps, x_0, h, 0.0, 0.85, 1.0, 0.85
            )


            x_hat = ((x_hat[-1] + 1) / 2).clamp(0, 1)      

            x_0_scaled = ((x_0.detach().cpu() + 1) / 2).clamp(0, 1)
            base_save_dir = "save_dir"
            save_dir = base_save_dir + f"test_results_b/steps_{num_steps}/image_{i}/"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_samples_torch(save_dir, x_hat)  

            x_hat_var = x_hat.var(dim=(0, 1), unbiased=True)
            x_hat_var = x_hat_var.clamp_min(1e-6)

            squared_error = (x_hat - x_0_scaled) ** 2       
            C_map = squared_error / x_hat_var               
            C = C_map.mean()

            gt_flat = x_0_scaled.view(-1)            
            se_flat = squared_error.view(-1)         

            bin_idx = torch.bucketize(gt_flat, bin_edges, right=False) - 1
            bin_idx = bin_idx.clamp(min=0, max=n_bins-1)

            pixel_stats[num_steps]["se_sum"].scatter_add_(0, bin_idx, se_flat)
            pixel_stats[num_steps]["count"].scatter_add_(0, bin_idx, torch.ones_like(se_flat, dtype=torch.long))



            mse = squared_error.mean()

            psnr = 10.0 * torch.log10(1.0 / mse)

            mean_save_dir = base_save_dir + f"test_results_b/steps_{num_steps}/means/"
            var_save_dir = base_save_dir + f"test_results_b/steps_{num_steps}/vars/"
            cal_save_dir = base_save_dir + f"test_results_b/steps_{num_steps}/cals/"
            Path(mean_save_dir).mkdir(parents=True, exist_ok=True)
            Path(var_save_dir).mkdir(parents=True, exist_ok=True)
            Path(cal_save_dir).mkdir(parents=True, exist_ok=True)

            mean_image_name = f"mean_{i}"
            var_image_name = f"var_{i}"
            cal_image_name = f"cal_{i}"

            x_hat_mean = x_hat.mean(dim=(0, 1))
            C_map_mean = C_map.mean(dim=(0, 1))

            save_single_sample_torch(mean_save_dir, mean_image_name, x_hat_mean)
            save_single_sample_torch(var_save_dir, var_image_name, torch.sqrt(x_hat_var))
            save_single_sample_torch(cal_save_dir, cal_image_name, C_map_mean)

            results.append({
                "image_idx": i,
                "num_steps": num_steps,
                "calibration": C.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
            })


results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)

rows = []
for num_steps in num_samples_step:
    se_sum = pixel_stats[num_steps]["se_sum"]
    count = pixel_stats[num_steps]["count"].clamp(min=1)

    mse_per_bin = se_sum / count

    for b in range(n_bins):
        rows.append({
            "num_steps": num_steps,
            "bin_low": float(bin_edges[b]),
            "bin_high": float(bin_edges[b + 1]),
            "mse": float(mse_per_bin[b]),
            "count": int(count[b]),
        })

pixel_mse_df = pd.DataFrame(rows)
pixel_mse_df.to_csv("pixel_mse_bins_first.csv", index=False)


                


    
