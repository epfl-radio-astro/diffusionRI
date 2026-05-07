import torch
from torch import nn
from .unet_parts import (
    PixelShuffleUpsampling,
    Up,
    Down,
    ResnetBlock,
)
from torch.nn import SiLU, GroupNorm, Conv2d
import math


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional encodings (DDPM style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class Unet(nn.Module):

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        depth: int,
        n_blocks: int,
        t_dim: int,
        concat: bool = True,
        activation=SiLU,
        normalization=GroupNorm,
        upsampling=PixelShuffleUpsampling,
    ):
        super().__init__()
        ch = [base_ch * 2**i for i in range(depth + 1)]

        self.in_conv = Conv2d(in_ch, base_ch, 3, 1, 6)

        self.out_conv = Conv2d(base_ch, in_ch, 1, 1, 0, bias=False)

        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.t_dim = t_dim

        self.enc = nn.ModuleList(
            [
                Down(ch[i], ch[i + 1], n_blocks, activation, normalization, t_dim)
                for i in range(depth)
            ]
        )

        self.bottleneck = ResnetBlock(ch[-1], ch[-1], activation, normalization, t_dim)
        print("Channel per encoder Depth")
        print(ch)
        self.dec = nn.ModuleList(
            [
                Up(
                    ch[i + 1],
                    ch[i],
                    n_blocks,
                    activation,
                    normalization,
                    upsampling,
                    t_dim,
                )
                for i in reversed(range(depth))
            ]
        )

    def forward(self, x, t):
        time_embeddings = timestep_embedding(t, self.t_dim)
        t_emb = self.time_mlp(time_embeddings)
        x = self.in_conv(x)

        skips = []
        for encode in self.enc:
            x, skip = encode(x, t_emb)

            skips.append((skip))

        x = self.bottleneck(x, t_emb)

        for decode, (skip) in zip(self.dec, reversed(skips)):
            x = decode(x, skip, t_emb)

        x = self.out_conv(x)

        return x[:, :, 5:-5, 5:-5]

