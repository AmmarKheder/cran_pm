import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class PixelShuffleUpBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.resblock = ResBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.act(x)
        return self.resblock(x)


class CNNDecoder(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        grid_h: int = 32,
        grid_w: int = 32,
        out_channels: int = 1,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_channels = out_channels

        in_ch = embed_dim + skip_channels if skip_channels > 0 else embed_dim
        if in_ch != embed_dim:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.skip_fuse = nn.Identity()

        channels = [embed_dim, 256, 128, 64, 32]
        self.stages = nn.ModuleList([
            PixelShuffleUpBlock(channels[i], channels[i + 1])
            for i in range(4)
        ])

        self.final = nn.Conv2d(channels[-1], out_channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, skip=None):
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.skip_fuse(x)

        for stage in self.stages:
            x = stage(x)

        return self.final(x)
