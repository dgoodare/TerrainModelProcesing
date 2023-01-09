
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim  # package implementing various optimisation algorithms
from torch.optim import lr_scheduler  # provides methods for adjusting the learning rate
from torch.utils.data import dataloader  # module for iterating over a dataset
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

class RMNet():
    def __init__(self, config):
        # image and mask dimensions
        self.imgWidth = config.imgWidth
        self.imgHeight = config.imgHeight
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

        # discriminator and generator
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)

        # initialise optimiser functions for Generator and Discriminator
        self.discOpt = optim.Adam()
        self.genOpt = optim.Adam()




