import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import pandas as pd
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
        maskPath = self.lookUp.iloc[index, 1]

        groundTruthDir = 'outputSlices/'
        maskDir = 'outputMasks/'

        # read the image and the mask
        groundTruth = torch.load(groundTruthDir + imgPath)
        mask = torch.load(maskDir + maskPath)

        return groundTruth, mask


def display_random_sample(dataset):
    """Create an example dataset and display a random sample from it"""

    sample1Idx = torch.randint(len(dataset), size=(1,)).item()
    originalImg1 = dataset[sample1Idx][0]
    mask1 = dataset[sample1Idx][1]

    originalImg2 = dataset[sample1Idx + 1][0]
    mask2 = dataset[sample1Idx + 1][1]

    originalImg3 = dataset[sample1Idx + 2][0]
    mask3 = dataset[sample1Idx + 2][1]

    f, ax = plt.subplots(3, 2)
    ax[0, 0].set_title("Ground truth")
    ax[0, 1].set_title("Mask")
    ax[0, 0].imshow(torch.Tensor.numpy(torch.squeeze(originalImg1)))
    ax[0, 1].imshow(torch.Tensor.numpy(torch.squeeze(mask1)))
    ax[1, 0].imshow(torch.Tensor.numpy(torch.squeeze(originalImg2)))
    ax[1, 1].imshow(torch.Tensor.numpy(torch.squeeze(mask2)))
    ax[2, 0].imshow(torch.Tensor.numpy(torch.squeeze(originalImg3)))
    ax[2, 1].imshow(torch.Tensor.numpy(torch.squeeze(mask3)))

    plt.show()
