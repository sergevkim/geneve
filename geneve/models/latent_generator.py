import einops
import torch
from torch.nn import (Conv2d, InstanceNorm2d, LeakyReLU, Linear, Module,
                      ModuleList, Parameter, Sequential)


class MappingNetwork(Module):
    def __init__(self, latent_dim: int, n_layers: int):
        super().__init__()

        layers_list = list()
        for _ in range(n_layers - 1):
            layers_list.append(Linear(latent_dim, latent_dim))
            layers_list.append(LeakyReLU(0.2))
        layers_list.append(Linear(latent_dim, latent_dim))
        self.layers = Sequential(*layers_list)

    def forward(self, x):
        return self.layers(x)


class AdaptiveInstanceNorm(Module):
    def __init__(self, in_channels: int, style_dim: int):
        super().__init__()
        self.norm = InstanceNorm2d(in_channels)
        self.style = Linear(style_dim, in_channels * 2)
        self.style.bias.data[:in_channels] = 1
        self.style.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out


class StyleBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int):
        super().__init__()
        self.conv_1 = \
            Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.adain_1 = AdaptiveInstanceNorm(out_channels, latent_dim)
        self.lrelu_1 = LeakyReLU(0.2)
        self.conv_2 = \
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.adain_2 = AdaptiveInstanceNorm(out_channels, latent_dim)
        self.lrelu_2 = LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv_1(x)
        x = self.adain_1(x, w)
        x = self.lrelu_1(x)
        x = self.conv_2(x)
        x = self.adain_2(x, w)
        x = self.lrelu_2(x)
        return x


class SynthesisNetwork(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        n_blocks: int,
        height: int,
        width: int,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.const_input = Parameter(torch.randn(in_channels, height, width))

        style_blocks_list = list()
        for _ in range(n_blocks - 1):
            style_blocks_list.append(
                StyleBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    latent_dim=latent_dim,
                )
            )
        style_blocks_list.append(
            StyleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                latent_dim=latent_dim,
            )
        )
        self.style_blocks = ModuleList(style_blocks_list)

    def forward(self, w, x=None):
        if x is None:
            bs = w.shape[0]
            x = einops.repeat(self.const_input, 'c h w -> bs c h w', bs=bs)
            noise = torch.randn(
                bs,
                self.in_channels,
                self.height,
                self.width,
            ).to(x.device)
            x = x + noise

        for block in self.style_blocks:
            x = block(x, w)

        return x


class LatentGenerator(Module):
    def __init__(
        self,
        height: int,
        width: int,
        latent_dim: int = 128,
        n_layers: int = 8,
        in_channels: int = 64,
        out_channels: int = 3,
        n_blocks: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping_network = \
            MappingNetwork(latent_dim=latent_dim, n_layers=n_layers)
        self.synthesis_network = SynthesisNetwork(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            n_blocks=n_blocks,
            height=height,
            width=width,
        )

    def forward(self, z):
        w = self.mapping_network(z)
        return self.synthesis_network(w)

    def generate_random(self):
        noise = torch.randn(8, self.latent_dim)
        return self.forward(noise)


if __name__ == '__main__':
    batch_size = 64

    adain = AdaptiveInstanceNorm(in_channels=3, style_dim=128)
    images = torch.randn(batch_size, 3, 32, 32)
    style = torch.randn(batch_size, 128)
    print('adain check: ', images.shape, style.shape)
    outs = adain(images, style)

    G = LatentGenerator(height=32, width=32)
    z = torch.randn(batch_size, 128)
    w = G.mapping_network(z)
    outs = G.synthesis_network(w)
    print('generator check: ', w.shape, outs.shape)
    print(G.generate_random().shape)

    print('====')
    print(outs[0,0,:3,:3])
    print(outs[1,0,:3,:3])
