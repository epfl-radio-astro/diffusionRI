import torch
import pytorch_lightning as pl
from real_valued.ddpm import DDPM
from torch_ema import ExponentialMovingAverage
from torch.optim import AdamW
from torch.nn import GroupNorm
from ddrm.svd_replacement import Fourier2D
import numpy as np
from helpers import save_tall_image_grid
from pathlib import Path


class LitDDPM(pl.LightningModule):

    def __init__(
        self,
        unet,
        timesteps=1000,
        lr=2e-4,
        batch_size=128,
        image_size=128,
        warmup_steps=0.05,
        max_steps=100_000,
        sigma_modifier=1.0,
        ema_decay=0.999,
        data_transformer=None,
        image_save_path="images/real_valued_images_large/",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["unet", "betas"])
        self.model = DDPM(unet, timesteps=timesteps)

        self.lr = lr
        self.image_size = image_size
        self.batch_size = batch_size
        self.total_steps = max_steps
        self.warmup_steps = warmup_steps * max_steps
        self.data_transformer = data_transformer
        self.ema = ExponentialMovingAverage(
            self.model.parameters(),
            decay=ema_decay,
            use_num_updates=True,
        )

        self.save_path = image_save_path
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    def _make_parameters(self):
        no_decay_classes = GroupNorm
        decay, no_decay = [], []
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if isinstance(module, no_decay_classes) or param_name.endswith("bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

        return [
            {"params": decay, "weight_decay": 1e-2},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        parameters = self._make_parameters()
        opt = AdamW(parameters, lr=self.lr, betas=(0.9, 0.999))

        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, total_iters=self.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_steps - self.warmup_steps, eta_min=5e-5
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()

    def export_ema(self, path: str):
        with self.ema.average_parameters():
            torch.save(self.model.state_dict(), path)

    def on_load_checkpoint(self, checkpoint):
        if "ema_state" in checkpoint:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.hparams.ema_decay if "ema_decay" in self.hparams else 0.999,
                use_num_updates=True,
            )
            self.ema.load_state_dict(checkpoint["ema_state"])

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    def ddpm_loss(self, x_pred, x_actual):
        return torch.nn.functional.mse_loss(x_pred, x_actual)

    def training_step(self, batch, batch_idx):

        x = torch.Tensor(batch)
        B = x.size(0)
        x = x.to(torch.float32)
        t = torch.randint(
            0, self.model.num_timesteps, (B,), device=self.device, dtype=torch.long
        )
        noise_pred, noise = self.model(x, t)

        noise_pred = noise_pred
        loss = self.ddpm_loss(noise_pred, noise)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        sch = self.lr_schedulers()
        self.log("lr", sch.get_last_lr()[0], prog_bar=True, on_step=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x_0 = batch.unsqueeze(1)
        with self.ema.average_parameters():
            mask = torch.from_numpy(np.load("uv_coverages/uv_sampling_60_steps.npy"))
            h = Fourier2D(
                channels=1, img_dim=self.image_size, S=mask, device=self.device
            )

            self.model.eval()
            x_hat, _ = self.model.sample_ddrm(
                self.image_size, 500, x_0, h, 0.025, 0.85, 1.0, 0.85
            )
            x_hat = ((x_hat[-1] + 1) / 2).clamp(0, 1)
            x_0 = ((x_0 + 1) / 2).clamp(0, 1)

            x_0 = x_0.to("cpu")

            mse = torch.mean((x_0 - x_hat).abs() ** 2)

            mae = torch.mean((x_hat - x_0).abs())
            psnr = -10 * torch.log10(mse + 1e-10)

        self.log("val/mse", mse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, on_epoch=True, sync_dist=True)
        self.log("val/psnr", psnr, prog_bar=True, on_epoch=True, sync_dist=True)

        save_tall_image_grid(
            x_hat, self.save_path + f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        )

        return mse
