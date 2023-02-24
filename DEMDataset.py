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
        maskPath = self.lookUp.iloc[index, 1]

        groundTruthDir = 'outputSlices/'
        maskDir = 'outputMasks/'

        # read the image and the mask
        groundTruth = np.load(groundTruthDir + imgPath)
        mask = np.load(maskDir + maskPath)
        # apply transformations
        gt_tens = self.transform(groundTruth)
        mask_tens = self.transform(mask)

        return gt_tens, mask_tens


def display_random_sample():
    """Create an example dataset and display a random sample from it"""

    dataset = DEMDataset('lookUpTable.csv', 'LookUp', transform=transforms.ToTensor())

    sampleIdx = torch.randint(len(dataset), size=(1,)).item()
    originalImg = dataset[sampleIdx][0]
    tl_e = dataset[sampleIdx][1]
    tr_e = dataset[sampleIdx][3]
    tl_s = dataset[sampleIdx][5]
    br_s = dataset[sampleIdx][7]
    c_s = dataset[sampleIdx][9]

    # print(f"original shape: {originalImg.shape}")

    f, ax = plt.subplots(2, 3)
    ax[0, 0].set_title(str(sampleIdx) + ": Ground truth")
    ax[0, 0].imshow(originalImg)
    ax[0, 1].imshow(tl_e)
    ax[0, 2].imshow(tr_e)
    ax[1, 0].imshow(tl_s)
    ax[1, 1].imshow(br_s)
    ax[1, 2].imshow(c_s)

    plt.show()
