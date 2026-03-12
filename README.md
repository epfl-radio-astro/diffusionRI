# diffusionRI



## Reproducing the experiment

### Runtime environment

We used the NVIDIA PyTorch container image 

[ngc-pytorch_25.08.sqsh](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html)

The dependencies list are provided in requirements-extra.txt

To recreate the environement run (inside the container image)

```
    python3 -m venv venv-pt-25.08 
    source venv-pt-25.08/bin/activate 
    pip install -r requirements-extra.txt
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