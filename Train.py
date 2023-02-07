import torch
import torch.nn as nn
import torch.optim as optim  # package implementing various optimisation algorithms
import torchvision
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader  # module for iterating over a dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

import DatasetManager
from DEMDataset import DEMDataset
from Models import Discriminator, Generator


import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random


def reverse_mask(x):
    """A function to reverse an image mask"""
    return 1-x


def select_mask_type():
    types = {
        1: 1,  # tl_edge
        2: 3,  # tr_edge
        3: 5,  # tl_strip
        4: 7,  # br_strip
        5: 9   # c_strip
    }
    i = random.randint(1, 5)
    return types[i]


def discriminator_loss(x, y):
    """
    Wasserstein loss function
    :param x: real
    :param y: fake
    :return:
    """
    return -(torch.mean(x) - torch.mean(y))


def generator_loss(x, y, disc_loss):
    """
    Loss function for generator network
    :param x: real
    :param y: fake
    :param disc_loss: the inverse of the loss of the discriminator
    :return:
    """
    # TODO: define a proper loss function
    input_mask = x[:, :, :, 3:]
    reversedMask = reverse_mask(input_mask)

    inputImg = x[:, :, :, 0:3]
    outputImg = y[:, :, :, 0:3]
    return disc_loss


# Define Hyper-parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learning_rate = 5e-5
Batch_size = 64
Img_Size = DatasetManager.img_size
Img_channels = 2
Z_dim = 100  # check what this actually is
Num_epochs = 1
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
        transforms.ToTensor(),
        # transforms.Normalize() - TODO: check with Iain if there are upper and lower limits for DEMs
    ]
)

# load the dataset
dataset = DEMDataset('lookUpTable.csv', rootDir='LookUp', transform=transforms)
Dataset_size = dataset.__len__()
print("Dataset loaded...")
print(f"Dataset size: {Dataset_size}")
# split into training and testing sets with an 80/20 ratio
trainingSet, testingSet = torch.utils.data.random_split(dataset, [int(Dataset_size*8/10), int(Dataset_size*2/10)])
print("Dataset split...")
# create dataloaders for each set
# TODO: look into multi-processing
trainingLoader = DataLoader(dataset=trainingSet, batch_size=Batch_size, shuffle=True)
print("training loader created...")
testingLoader = DataLoader(dataset=testingSet, batch_size=Batch_size, shuffle=True)
print("testing loader created...")


# Initialise the two networks
gen = Generator(Z=Z_dim, imgChannels=Img_channels, features=Features_gen).to(device)
print("generator initialised...")
disc = Discriminator(imgChannels=Img_channels, features=Features_disc).to(device)
print("discriminator initialised...")
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
print("optimisers defined...")

# Define random noise to being training with
fixed_noise = torch.randn(32, Img_channels, 1, 1).to(device)

# Data Visualisation stuff
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
print("Summary writers created...")

step = 0
gen.train()
disc.train()
print(gen)
print("ready to train...")


for epoch in range(Num_epochs):
    for batch_idx, sample in enumerate(trainingLoader):
        print(f"Batch index: {batch_idx}")
        # choose which mask type to use for this epoch
        maskID = select_mask_type()
        # retrieve ground truth, masked image, and corresponding mask
        real = sample[0].to(device)
        maskedImg = sample[maskID].to(device)
        mask = sample[maskID+1].to(device)
        print(f"Real size: {real.shape}")
        print(f"Masked Img size: {maskedImg.shape}")
        print(f"Mask size: {mask.shape}")

        # train discriminator
        for _ in range(Disc_iters):
            noise = torch.randn((Batch_size, Z_dim, 1, 1)).to(device)
            # TODO: make sure masks are passed properly as parameters
            fake = gen(x=noise, maskedImg=maskedImg, mask=mask)
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
        loss_gen = generator_loss(real, fake, -torch.mean(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # display results at specified intervals
        if batch_idx % 100 == 0:
            print(
                f"========================================="
                f"|| Epoch [{epoch}/{Num_epochs}] -- Batch [{batch_idx}/{len(trainingLoader)}]"
                f"|| Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # pick up to 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
