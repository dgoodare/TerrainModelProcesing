
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

print(cv2.__version__)

def ReverseMask(img):
    """A function to reverse an image mask"""
    return 1-img

class RMNet():
    def __init__(self, config):
        # image and mask dimensions
        self.imgSize = config.imgSize
        self.imgChannels = config.imgChannels
        self.maskChannels = config.maskChannels
        self.imgShape = (self.imgWidth, self.imgHeight, self.imgChannels)
        self.maskShape = (self.imgWidth, self.imgHeight, self.maskChannels)
        self.missingShape = (self.imgWidth, self.imgHeight, self.imgChannels)

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
        imageTransforms = transforms.Compose(
            [
                transforms.Resize(self.imgSize),
                transforms.ToTensor(),
                # transforms.Normalize()
            ]
        )

        # create and load dataset
        self.dataset = datasets.ImageFolder(root=self.imgDir, transform=imageTransforms)
        self.loader = DataLoader(dataset=self.dataset, batch_size=self.batchSize, shuffle=True)

        # Initialise the discriminator network
        self.discriminator = Discriminator(features=config.discFeatures, imgChannels=self.imgChannels)

        # Initialise the generator network, first create the reverse-masked image

        self.generator = Generator(features=config.genFeatures, imgChannels=self.imgChannels)

        # initialise optimiser functions for Generator and Discriminator
        self.discOpt = optim.Adam()
        self.genOpt = optim.Adam()

