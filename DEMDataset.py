import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np


class DEMDataset(Dataset):
    """A modified version of the PyTorch Dataset class"""
    def __init__(self, csvFile, rootDir, transform=None):
        self.lookUp = pd.read_csv(rootDir + '/' + csvFile)
        self.rootDir = rootDir
        self.transform = transform

    def __len__(self):
        """Returns the number of unmasked images"""
        return len(self.lookUp)

    def __getitem__(self, index):
        # get the names of the files in row 'index'
        imgPath = self.lookUp.iloc[index, 0]
        squareImgPath = self.lookUp.iloc[index, 1]
        squareMaskPath = self.lookUp.iloc[index, 2]
        stripImgPath = self.lookUp.iloc[index, 3]
        stripMaskPath = self.lookUp.iloc[index, 4]

        groundTruthDir = 'slicedImages/'
        maskedImagesDir = 'maskedImages/'
        maskDir = 'outputMasks/'

        # read the image
        groundTruth = Image.open(groundTruthDir + imgPath)
        # get masked images and their corresponding masks and convert them to tensor objects
        squareImg = Image.open(maskedImagesDir + squareImgPath)
        squareMask = np.load(maskDir + squareMaskPath)
        stripImg = Image.open(maskedImagesDir + stripImgPath)
        stripMask = np.load(maskDir + stripMaskPath)

        # apply transformations if they have been specified
        if self.transform:
            squareImg = self.transform(squareImg)
            squareMask = self.transform(squareMask)
            stripImg = self.transform(stripImg)
            stripMask = self.transform(stripMask)

        return groundTruth, squareImg, squareMask, stripImg, stripMask


dataset = DEMDataset('lookUpTable.csv', 'LookUp',)

figure, axs = plt.subplots(nrows=3, figsize=(3, 3))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sampleIdx = torch.randint(len(dataset), size=(1,)).item()
    originalImg = dataset[sampleIdx][0]
    square = dataset[sampleIdx][1]
    strip = dataset[sampleIdx][3]

    axs[0].set_title(str(sampleIdx) + " Ground truth")
    axs[0].imshow(originalImg)
    axs[1].set_title(str(sampleIdx) + " Square mask")
    axs[1].imshow(square)
    axs[2].set_title(str(sampleIdx) + " Strip mask")
    axs[2].imshow(strip)

plt.show()
