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
        tl_e_ImgPath = self.lookUp.iloc[index, 1]
        tl_e_MaskPath = self.lookUp.iloc[index, 2]
        tr_e_ImgPath = self.lookUp.iloc[index, 3]
        tr_e_MaskPath = self.lookUp.iloc[index, 4]
        tl_s_ImgPath = self.lookUp.iloc[index, 5]
        tl_s_MaskPath = self.lookUp.iloc[index, 6]
        br_s_ImgPath = self.lookUp.iloc[index, 7]
        br_s_MaskPath = self.lookUp.iloc[index, 8]
        c_s_ImgPath = self.lookUp.iloc[index, 9]
        c_s_MaskPath = self.lookUp.iloc[index, 10]

        groundTruthDir = 'slicedImages/'
        maskedImagesDir = 'maskedImages/'
        maskDir = 'outputMasks/'

        # read the image
        groundTruth = Image.open(groundTruthDir + imgPath)
        # get masked images and their corresponding masks and convert them to tensor objects
        tl_e_img = Image.open(maskedImagesDir + tl_e_ImgPath)
        tl_e_mask = np.load(maskDir + tl_e_MaskPath)
        tr_e_img = Image.open(maskedImagesDir + tr_e_ImgPath)
        tr_e_mask = np.load(maskDir + tr_e_MaskPath)
        tl_s_img = Image.open(maskedImagesDir + tl_s_ImgPath)
        tl_s_mask = np.load(maskDir + tl_s_MaskPath)
        br_s_img = Image.open(maskedImagesDir + br_s_ImgPath)
        br_s_mask = np.load(maskDir + br_s_MaskPath)
        c_s_img = Image.open(maskedImagesDir + c_s_ImgPath)
        c_s_mask = np.load(maskDir + c_s_MaskPath)

        # apply transformations if they have been specified
        if self.transform:
            tl_e_img = self.transform(tl_e_img)
            tl_e_mask = self.transform(tl_e_mask)
            tr_e_img = self.transform(tr_e_img)
            tr_e_mask = self.transform(tr_e_mask)
            tl_s_img = self.transform(tl_s_img)
            tl_s_mask = self.transform(tl_s_mask)
            br_s_img = self.transform(br_s_img)
            br_s_mask = self.transform(br_s_mask)
            c_s_img = self.transform(c_s_img)
            c_s_mask = self.transform(c_s_mask)

        return groundTruth, tl_e_img, tl_e_mask, tr_e_img, tr_e_mask, tl_s_img, tl_s_mask, br_s_img, br_s_mask, c_s_img, c_s_mask


dataset = DEMDataset('lookUpTable.csv', 'LookUp',)

figure, axs = plt.subplots(nrows=6, figsize=(6, 6))
cols, rows = 6, 6
for i in range(1, cols * rows):
    sampleIdx = torch.randint(len(dataset), size=(1,)).item()
    originalImg = dataset[sampleIdx][0]
    tl_e = dataset[sampleIdx][1]
    tr_e = dataset[sampleIdx][3]
    tl_s = dataset[sampleIdx][5]
    br_s = dataset[sampleIdx][7]
    c_s = dataset[sampleIdx][9]

    axs[0].set_title(str(sampleIdx) + ": Ground truth")
    axs[0].imshow(originalImg)
    axs[1].imshow(tl_e)
    axs[2].imshow(tr_e)
    axs[3].imshow(tl_s)
    axs[4].imshow(br_s)
    axs[5].imshow(c_s)

plt.show()
