import torch
import torch.nn as nn


class SimpleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(128, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    batch_size = 64
    noise = torch.randn(batch_size, 128, 1, 1)
    g = SimpleGenerator()
    print(g(noise).shape)

