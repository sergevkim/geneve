import torch
from einops.layers.torch import Rearrange
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, LeakyReLU,
                      Linear, Module, Sequential, Sigmoid)


class ConvBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, act: bool = None):
        super().__init__()
        self.act = act
        self.conv = \
            Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.norm = BatchNorm2d(out_channels)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.lrelu(x)
        return x


class Discriminator(Module):
    def __init__(self, n_channels: int = 64, n_blocks: int = 8):
        super().__init__()
        self.rgb_block = ConvBlock(3, n_channels)
        self.blocks = Sequential(*[
            ConvBlock(n_channels, n_channels) for i in range(n_blocks)
        ])
        self.neck = Sequential(
            AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('bs c 1 1 -> bs c'),
        )
        self.head = Sequential(
            Linear(n_channels, 1, bias=True),
            Sigmoid(),
        )

    def forward(self, x):
        x = self.rgb_block(x)
        x = self.blocks(x)
        x = self.neck(x)
        x = self.head(x)

        return x

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 3
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            Rearrange('bs 1 1 1 -> bs 1')
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    D = Discriminator()
    images = torch.randn(8, 3, 32, 32)
    outs = D(images)
    print(outs.shape)