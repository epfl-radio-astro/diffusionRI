# diffusionRI



## Reproducing the experiment

### Runtime environment

#### Option A: Conda (standalone, recommended for most users)

Create a new conda environment with CUDA-enabled PyTorch:

```bash
conda create -n diffusionRI python=3.11 -y
conda activate diffusionRI

# Install PyTorch with CUDA support (adjust cu128 to match your CUDA version, e.g. cu121, cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install pytorch-lightning torch-ema astropy h5py matplotlib numpy pandas Pillow tensorboard
```

To check your CUDA version: `nvidia-smi | grep "CUDA Version"`

#### Option B: NGC container (original environment)

We used the NVIDIA PyTorch container image

[ngc-pytorch_25.08.sqsh](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html)

The full container dependency list is provided in `extra-requirements.txt`.

To recreate the environment run (inside the container image):

```bash
python3 -m venv venv-pt-25.08
source venv-pt-25.08/bin/activate
pip install -r extra-requirements.txt
```


### Training the model

From here run :

``` python train_real_valued.py```

or on a slurm system :

``` srun -ul --environment=./cscs/ngc-pytorch-25.08.toml bash -c '
    source "$HOME/venv-pt-25.08/bin/activate"
    python train_real_valued.py
'
```

On a first run this will use internet connection to download the training dataset

### Running inference



``` python compute_stats_first.py ```

or on a slurm system :

``` srun -ul --environment=./cscs/ngc-pytorch-25.08.toml bash -c '
    source "$HOME/venv-pt-25.08/bin/activate"
    python compute_stats_first.py
'
```