import torch.nn as nn


class Discriminator(nn.Module):
    """A class to represent a discriminator within a GAN"""

    def __init__(self, imgChannels,  features):
        super(Discriminator, self).__init__()
        self.features = features

        # Layers of the discriminator network
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=imgChannels, out_channels=imgChannels, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(features * 2, out_channels=imgChannels, kernel_size=3, stride=2, padding=0),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(num_features=features, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
        )

        # layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(features * 4, out_channels=imgChannels, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=features, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
        )

        # layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(features * 8, out_channels=imgChannels, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=features, momentum=0.8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = nn.Flatten()(output)
        output = nn.Linear(self.features, 1)(output)

        return output


class Generator(nn.Module):
    """A class to represent a generator within a GAN"""

    def __init__(self, imgChannels, features):
        super(Generator, self).__init__()

        ###
        # Encoding Stage:
        #   This stage encodes image features in latent space. Each layer consists of a block of convolution, with
        #   a 5x5 kernel size, an output size of 64, and a dilation rate of 2. The purpose of dilation is to capture
        #   finer details as well as textural information.
        #   The MaxPool2D function is used to down-sample the image in order to reduce variance and computational
        #   complexity.
        ###
        self.encodeLayer = nn.Sequential(
            nn.Conv2d(in_channels=imgChannels, out_channels=features, kernel_size=(5, 5), dilation=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=features, momentum=0.8)
        )

        # the final encoding layer includes dropout instead of batch normalisation to prevent over-fitting
        self.encodeLayerDropout = nn.Sequential(
            nn.Conv2d(in_channels=imgChannels, out_channels=features, kernel_size=(5, 5), dilation=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout()
        )

        ###
        # Decoding stage:
        #
        ###

        self.decodeLayer = nn.Sequential(
            nn.Upsample(size=(2, 2), mode='bilinear'),
            nn.ConvTranspose2d(in_channels=imgChannels, out_channels=features, kernel_size=(5, 5), ),
            # Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))
            nn.ReLU(),
            nn.BatchNorm2d(num_features=features, momentum=0.8)
        )
        self.outputLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=imgChannels, out_channels=features, kernel_size=(3, 3)),
            nn.Tanh()
        )

    def forward(self, x):
        # encode
        output = self.encodeLayer(x)
        pool1 = nn.MaxPool2d(2, 2)(output)

        output = self.encodeLayer(pool1)
        pool2 = nn.MaxPool2d(2, 2)(output)
        self.features *= 2

        output = self.encodeLayer(pool2)
        pool3 = nn.MaxPool2d(2, 2)(output)
        self.features *= 2

        output = self.encodeLayer(pool3)
        pool4 = nn.MaxPool2d(2, 2)(output)
        self.features *= 2
        output = self.encodeLayerDropout(pool4)

        # decode
        output = self.decodeLayer(output)
        self.features /= 2

        output = self.decodeLayer(output)
        self.features /= 2

        output = self.decodeLayer(output)
        self.features /= 2

        output = self.decodeLayer(output)
        output = self.outputLayer(output)

        return output
