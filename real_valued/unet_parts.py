import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, Identity, SiLU, GroupNorm


class PixelShuffleUpsampling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        normalization,
        embed_dim: int | None = None,
    ):
        super().__init__()

        self.skip = (
            Identity()
            if in_channels == out_channels
            else Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.bn1 = normalization(8, out_channels)
        self.bn2 = normalization(8, out_channels)

        self.act1 = activation()
        self.act2 = activation()

        self.with_embed = embed_dim is not None
        if self.with_embed:
            self.film1 = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim, 2 * out_channels), nn.Dropout(0.1)
            )

            self.film2 = nn.Sequential(
                nn.SiLU(), nn.Linear(embed_dim, 2 * out_channels), nn.Dropout(0.1)
            )

        self.scale = 1 / math.sqrt(2)

    def forward(self, x, t_embed=None):
        skip = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)

        if self.with_embed:
            if t_embed is None:
                raise ValueError("t_emb must be provided when embed_dim is set")

            film = self.film1(t_embed)

            (
                gamma,
                beta,
            ) = film.chunk(2, dim=1)

            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            x = gamma * x + beta

        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.with_embed:
            if t_embed is None:
                raise ValueError("t_emb must be provided when embed_dim is set")

            film = self.film2(t_embed)

            (
                gamma,
                beta,
            ) = film.chunk(2, dim=1)

            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            x = gamma * x + beta

        x = (x + skip) * self.scale
        x = self.act2(x)

        return x


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=2,
        activation=SiLU,
        normalization=GroupNorm,
        embed_dim=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResnetBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    activation,
                    normalization,
                    embed_dim,
                )
                for i in range(num_blocks)
            ]
        )

        self.downsample = Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)

    def forward(self, x, t_embed=None):
        skip = x
        for blk in self.blocks:
            x = blk(x, t_embed)
        x = self.downsample(x)
        return x, skip


class Up(nn.Module):
    def __init__(
        self,
        channels_encoder,
        channels_skip,
        num_blocks=2,
        activation=SiLU,
        normalization=GroupNorm,
        upsample=PixelShuffleUpsampling,
        embed_dim=None,
    ):
        super().__init__()
        self.up = upsample(channels_encoder)

        in_ch = channels_encoder + channels_skip
        out_ch = channels_skip
        self.blocks = nn.ModuleList(
            [
                ResnetBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    activation=activation,
                    normalization=normalization,
                    embed_dim=embed_dim,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x, skip, t_embed=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        for blk in self.blocks:
            x = blk(x, t_embed)
        return x
