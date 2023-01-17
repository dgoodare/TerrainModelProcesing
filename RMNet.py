
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim  # package implementing various optimisation algorithms
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import DataLoader  # module for iterating over a dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

from Models import Discriminator, Generator

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy


def ReverseMask(img):
    """A function to reverse an image mask"""
    return 1-img


class RMNet:
    def __init__(self, config):
        # image and mask dimensions
        self.imgSize = config.imgSize
        self.imgChannels = config.imgChannels
        self.maskChannels = config.maskChannels
        self.imgShape = (self.imgSize, self.imgSize, self.imgChannels)
        self.maskShape = (self.imgSize, self.imgSize, self.maskChannels)
        self.missingShape = (self.imgSize, self.imgSize, self.imgChannels)

        # training Hyper-parameters
        self.numEpochs = config.numEpochs
        self.batchSize = config.batchSize
        self.startTime = time.time()
        self.endTime = time.time()
        self.sampleInterval = config.sampleInterval
        self.currentEpoch = config.currentEpoch
        self.lastTrainedEpoch = config.lastTrainedEpoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # file and folder variables
        self.datasetName = 'RMNet_dataset'
        self.modelsPath = 'models'
        self.imgDir = r'/images/train/celebA'
        self.maskDir = r'/masks/train/qd'
        self.imgsInPath = os.listdir(self.imgDir)
        self.masksInPath = os.listdir(self.maskDir)

        self.continueTrain = True

        # set transformation for images to make sure they are the correct size
        # and then convert them to tensors
        imageTransforms = transforms.Compose(
            [
                transforms.Resize(self.imgSize),
                transforms.ToTensor(),
                # transforms.Normalize()
            ]
        )

        # create and load dataset
        self.dataSet = datasets.ImageFolder(root=self.imgDir, transform=imageTransforms)
        self.dataLoader = DataLoader(dataset=self.dataSet, batch_size=self.batchSize, shuffle=True)

        # create and load mask set
        self.maskSet = datasets.ImageFolder(root=self.maskDir, transform=imageTransforms)
        self.maskLoader = DataLoader(dataset=self.maskSet, batch_size=self.batchSize, shuffle=True)

        # Initialise the discriminator network
        self.discriminator = Discriminator(features=config.discFeatures, imgChannels=self.imgChannels)

        # Initialise the generator network
        # first create the reverse-masked image
        self.generator = Generator(features=config.genFeatures, imgChannels=self.imgChannels)

        # initialise optimiser functions for Generator and Discriminator
        self.discOpt = optim.Adam(params=self.discriminator.parameters(),
                                  lr=config.discLR,
                                  betas=(config.beta1, config.beta2),
                                  eps=config.epsilon)
        self.genOpt = optim.Adam(params=self.generator.parameters(),
                                 lr=config.genLR,
                                 betas=(config.beta1, config.beta2),
                                 eps=config.epsilon)

    def buildGAN(self, discriminator, generator):
        if torch.cuda.is_available():
            image = torch.cuda.ByteTensor(self.imgSize, self.imgSize, self.imgChannels)
            mask = torch.cuda.ByteTensor(self.imgSize, self.imgSize, self.imgChannels)
        else:
            image = torch.ByteTensor(self.imgSize, self.imgSize, self.imgChannels)
            mask = torch.ByteTensor(self.imgSize, self.imgSize, self.imgChannels)

        # generator creates fake image
        genOutput = generator([image, mask])

        # train generator only for the combined model

        # discriminator assesses the generated image
        genImg = (lambda x: x[:, :, :, 0:3])(genOutput)
        score = discriminator(genImg)

        # return





