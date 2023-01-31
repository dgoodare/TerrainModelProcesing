import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np


class DEMDataset(Dataset):
    """A modified version of the PyTorch Dataset class"""
    def __init__(self, csvFile, rootDir, transform=None):
        self.lookUp = pd.read_csv(csvFile)
        self.rootDir = rootDir
        self.transform = transform

    def __len__(self):
        """Returns the number of unmasked images"""
        return len(self.lookUp)

    def __getitem__(self, index):
        # get the names of the files in row 'index'
        imgPath = os.path.join(self.rootDir, self.lookUp.iloc[index, 0])
        squareImgPath = os.path.join(self.rootDir, self.lookUp.iloc[index, 1])
        squareMaskPath = os.path.join(self.rootDir, self.lookUp.iloc[index, 2])
        stripImgPath = os.path.join(self.rootDir, self.lookUp.iloc[index, 3])
        stripMaskPath = os.path.join(self.rootDir, self.lookUp.iloc[index, 4])

        groundTruthDir = 'slicedImages/'
        maskedImagesDir = 'maskedImages/'
        maskDir = 'outputMasks/'

        # read the image
        groundTruth = Image.open(groundTruthDir + imgPath)
        # get masked images and their corresponding masks and convert them to tensor objects
        squareImg = torch.tensor(Image.open(maskedImagesDir + squareImgPath))
        squareMask = torch.tensor(np.load(maskDir + squareMaskPath))
        stripImg = torch.tensor(Image.open(maskedImagesDir + stripImgPath))
        stripMask = torch.tensor(np.load(maskDir + stripMaskPath))

        # apply transformations if they have been specified
        if self.transform:
            squareImg = self.transform(squareImg)
            squareMask = self.transform(squareMask)
            stripImg = self.transform(stripImg)
            stripMask = self.transform(stripMask)

        return groundTruth, squareImg, squareMask, stripImg, stripMask
