import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import filters, feature
from skimage.color import rgb2gray
import os
import torch
import random
from torchmetrics.image.fid import FrechetInceptionDistance


def DetectEdges(directory):
    """Highlights prominent edges in images in a given directory"""
    originalList = []
    sobelList = []
    prewittList = []
    cannyList = []
    num_images = 5
    count = 0
    dirLength = len(os.listdir(directory))
    rand_idx = random.randrange(dirLength - num_images)

    for file in os.listdir(directory)[rand_idx:rand_idx+num_images]:
        tensor = torch.load(directory + "/" + file)
        tensor = torch.squeeze(tensor)
        img = torch.Tensor.numpy(tensor)

        prewitt = filters.prewitt(img)
        canny = feature.canny(img)

        originalList.append(img)
        prewittList.append(prewitt)
        cannyList.append(canny)
        count += 1
        if count > num_images:
            break

    f, ax = plt.subplots(num_images, 3)
    ax[0, 0].set_title("Original")
    ax[0, 1].set_title("Prewitt")
    ax[0, 2].set_title("Canny")
    for idx in range(num_images):
        ax[idx, 0].imshow(originalList[idx])
        ax[idx, 0].axis('off')
        ax[idx, 1].imshow(prewittList[idx])
        ax[idx, 1].axis('off')
        ax[idx, 2].imshow(cannyList[idx])
        ax[idx, 2].axis('off')
    plt.show()


def CalculateFID():
    fid = FrechetInceptionDistance(feature=64)
    realDir = ''
    fakeDir = ''
    real = FolderToTensor(realDir)
    fake = FolderToTensor(fakeDir)

    fid.update(real, real=True)
    fid.update(fake, real=False)
    fid.compute()


def FolderToTensor(folder):
    t = torch.Tensor()
    for file in os.listdir(folder):
        f = torch.load(file)
        t = torch.concat([t, f], dim=1)

    return t


DetectEdges("outputSlices")

