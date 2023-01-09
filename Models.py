import torch
import torch.nn as nn
from torchvision import transforms


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.imgChannels = config.imgChannels
        self.maskChannels = config.maskChannels
        self.missingShape = config.missingShape
        self.features = config.discFeatures

        self.disc = nn.Sequential(
            nn.Conv2d(self.missingShape, self.features, kernel_size=3, strides=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(self.features * 2, kernel_size=3, strides=2, padding=0),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(self.features * 4, kernel_size=3, strides=2, padding=0),
            nn.BatchNorm2d(momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(self.features * 8, kernel_size=3, strides=2, padding=0),
            nn.BatchNorm2d(momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(1)
        )

    def forward(self, x):
        return self.disc(x)


def reversemask(x):
    return 1-x


class Generator(nn.Module):
    def __init__(self, config):
        self.inputImg = torch.tensor(shape="config.imgShape", dtype="float32", name="imageInput")
        self.inputMask = torch.tensor(shape="config.maskShape", dtype="float32", name="maskInput")
        self.reversedMask = transforms.Lambda(reversemask(self.inputMask))
        self.maskedImg = torch.mul(self.inputImg, self.reversedMask)


