import torch
import torch.nn as nn
import torch.optim as optim  # package implementing various optimisation algorithms
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader  # module for iterating over a dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

from DEMDataset import DEMDataset
from Models import Discriminator, Generator

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


def reverse_mask(mask):
    """A function to reverse an image mask"""
    return 1-mask


def discriminator_loss(x, y):
    """
    Wasserstein loss function
    :param x: real
    :param y: fake
    :return:
    """
    return -(torch.mean(x) - torch.mean(y))


def generator_loss(x, y):
    """
    Loss function for generator network
    :param x: real
    :param y: fake
    :return:
    """
    mask = x[:, :, :, 3:]
    reversedMask = reverse_mask(mask)

    inputImg = x[:, :, :, 0:3]
    outputImg = y[:, :, :, 0:3]
    pass


# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 5e-5
Batch_size = 64
Img_Size = 64  # include function to determine this automatically
Img_channels = 2
Z_dim = 100  # check what this actually is
Num_epochs = 5
Features_disc = 64
Features_gen = 64
Disc_iters = 5
Weight_clip = 0.01  # check what this is too
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08

# transformations applied to datasets
transforms = transforms.Compose(
    [
        transforms.Resize(Img_Size),
        transforms.ToTensor(),
        # transforms.Normalize() - check with Iain if there are upper and lower limits for DEMs
    ]
)

# load the dataset
dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
Dataset_size = dataset.__len__()
# split into training and testing sets with an 80/20 ratio
trainingSet, testingSet = torch.utils.data.random_split(dataset, [(Dataset_size*8/10), (Dataset_size*2/10)])
# create dataloaders for each set
trainingLoader = DataLoader(dataset=trainingSet, batch_size=Batch_size, shuffle=True)
testingLoader = DataLoader(dataset=testingSet, batch_size=Batch_size, shuffle=True)


# Initialise the two networks
gen = Generator(imgChannels=Img_channels, features=Features_gen)
disc = Discriminator(imgChannels=Img_channels, features=Features_disc)
# initialise weights

# Optimiser Functions
opt_gen = optim.Adam(params=gen.parameters(),
                     lr=Learning_rate,
                     betas=(beta1, beta2),
                     eps=epsilon)
opt_disc = optim.Adam(params=disc.parameters(),
                      lr=Learning_rate,
                      betas=(beta1, beta2),
                      eps=epsilon)

# Define random noise to being training with
fixed_noise = torch.randn(32, Z_dim, 1, 1).to(device)

# Data Visualisation stuff
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0
gen.train()
disc.train()


for epoch in range(Num_epochs):
    for batch_idx, (real, _) in enumerate(trainingLoader):
        print(f"Batch index: {batch_idx}")
        real = real.to(device)

        # train discriminator
        for _ in range(Disc_iters):
            noise = torch.randn((Batch_size, Z_dim, 1, 1)).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            loss_disc = discriminator_loss(disc_real, disc_fake)
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()

            for p in disc.parameters():
                p.data.clamp_(-Weight_clip, Weight_clip)

        # train generator
        output = disc(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # display results at specified intervals
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{Num_epochs}] --- Batch [{batch_idx}/{len(trainingLoader)}]"
                f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # pick up to 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
