import torch

torch.set_float32_matmul_precision("high")
from real_valued.trainer import LitDDPM
from real_valued.unet import Unet
from pytorch_lightning import Trainer
from dataset.dataset import train_dataset
from helpers import first_radio_galaxy_transform_val
from torch.utils.data import DataLoader
from dataset.radio_galaxy_dataset.firstgalaxydata import FIRSTGalaxyData
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print("Current device:", idx)
    print("Name:", torch.cuda.get_device_name(idx))
    # quick sanity: actually allocate on it
    x = torch.randn(1, device=f"cuda:{idx}")
    print("Tensor device:", x.device)

import os

idx = torch.cuda.current_device()
props = torch.cuda.get_device_properties(idx)

print("Logical index:", idx)
print("Name:", props.name)
print("PCI bus ID:", props.pci_bus_id)
print("UUID:", props.uuid if hasattr(props, "uuid") else "N/A")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


batch_size = 10
training_steps = 100

image_size = 150
model = Unet(1, 32, 4, 2, 256).to("cuda")

## Doing one forward pass to initialize lazy variables
t = torch.randint(0, 200, (batch_size,), device=torch.device("cuda"), dtype=torch.long)
model(
    torch.randn(batch_size, 1, image_size, image_size, device=torch.device("cuda")), t
)
print("forward pass done")


ds = train_dataset()

dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=15,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

val_data = FIRSTGalaxyData(
    root="./data/",
    selected_split="valid",
    selected_classes=["FRI", "FRII", "Compact", "Bent"],
    input_data_list=["galaxy_data_h5.h5"],
    is_PIL=False,
    is_RGB=True,
    transform=first_radio_galaxy_transform_val,
)

val_dl = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)

trainer_model = LitDDPM(
    model,
    timesteps=1000,
    lr=1e-4,
    warmup_steps=0.05,
    max_steps=training_steps,
    image_size=image_size,
    ema_decay=0.999,
)


ckpt = ModelCheckpoint(
    dirpath="checkpoints",
    filename="step={step}-epoch={epoch}_run_imag_norm",
    every_n_train_steps=10_000,
    save_on_train_epoch_end=False,
    save_last=True,
)

logger = TensorBoardLogger(
    save_dir=str("experiments"),
    name="real_valued",
)

trainer = Trainer(
    max_steps=training_steps,
    accelerator="gpu",
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    max_epochs=-1,
    log_every_n_steps=100,
    callbacks=[ckpt],
    enable_progress_bar=False,
    val_check_interval=10_000,
    check_val_every_n_epoch=None,
    devices=1,
    logger=logger,
    precision="bf16-mixed", ## Change this if gpu does not support bf16
)

trainer.fit(trainer_model, dl, val_dl)

print("TRAINING DONE")


ddpm = trainer.model.model
ddpm.eval()
ddpm_c = ddpm.to("cuda")


