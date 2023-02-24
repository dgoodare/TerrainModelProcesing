import torch
import torch.nn as nn
from CreateDataset import img_size


class Discriminator(nn.Module):
    """
    A class to represent a discriminator within a GAN
    -   The design of the network follows the architecture described in 'Void Filling of Digital Elevation Models with
        a Terrain Texture Learning Model Based on Generative Adversarial Networks' (Qui et al. 2019)
    """

    def __init__(self, imgChannels,  features):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(imgChannels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.block(features, features * 2, 4, 2, 1),
            self.block(features * 2, features * 4, 4, 2, 1),
            self.block(features * 4, features * 8, 4, 2, 1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0)
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    """
    A class to represent a generator within a GAN
    -   The design of the network follows the architecture described in 'Void Filling of Digital Elevation Models with
        a Terrain Texture Learning Model Based on Generative Adversarial Networks' (Qui et al. 2019)
    """

    def __init__(self, Z, imgChannels, features):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            self.block(Z, features * 16, 4, 1, 0),
            self.block(features * 16, features * 8, 4, stride=1, padding=1),
            self.block(features * 8, features * 4, 4, stride=1, padding=1),
            self.block(features * 4, features * 2, 4, stride=1, padding=1),
            nn.Upsample(size=(img_size, img_size), mode='nearest'),
            nn.Conv2d(features * 2, imgChannels, 5, 1, padding='same'),
            nn.Tanh(),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Upsample(size=(img_size, img_size), mode='nearest'),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


def initialise_weights(model):
    """
    Initialises the weights for a nn model
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):  # TODO: check if these are correct
            nn.init.normal_(module.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 1, 64, 64
    z_dim = 100
    z = torch.randn((N, z_dim, 1, 1))
    x = torch.randn((N, in_channels, H, W))

    gen = Generator(z_dim, in_channels, 8)
    initialise_weights(gen)

    assert gen(z).shape == (N, in_channels, H, W)
    print("Generator created...")

    disc = Discriminator(in_channels, N)
    initialise_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    print("Discriminator created...")


# test()
