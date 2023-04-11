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
    prewittList = []
    cannyList = []
    harrisList = []
    num_images = 5
    count = 0
    # dirLength = len(os.listdir(directory))
    # rand_idx = random.randrange(dirLength - num_images)

    for file in os.listdir(directory):  # [rand_idx:rand_idx+num_images]:
        tensor = torch.load(directory + "/" + file).cpu()

        tensor = torch.squeeze(tensor)
        img = torch.Tensor.numpy(tensor)

        # prewitt edge detection
        prewitt = filters.prewitt(img)
        # canny edge detection
        canny = feature.canny(img)
        # harris corner detection
        harris = feature.corner_harris(img, k=0.01)
        peaks = feature.corner_peaks(harris, min_distance=5)
        subpix = feature.corner_subpix(img, peaks, window_size=64)

        originalList.append(img)
        prewittList.append(prewitt)
        cannyList.append(canny)
        harrisList.append(subpix)

        count += 1
        if count > num_images:
            break

    f, ax = plt.subplots(num_images, 4)
    ax[0, 0].set_title("Original")
    ax[0, 1].set_title("Prewitt")
    ax[0, 2].set_title("Canny")
    ax[0, 3].set_title("Harris Corner")
    for idx in range(num_images):
        ax[idx, 0].imshow(originalList[idx])
        ax[idx, 0].axis('off')
        ax[idx, 1].imshow(prewittList[idx])
        ax[idx, 1].axis('off')
        ax[idx, 2].imshow(cannyList[idx])
        ax[idx, 2].axis('off')
        ax[idx, 3].imshow(originalList[idx], interpolation='nearest')
        ax[idx, 3].plot(peaks[:, 1], peaks[:, 0], '+r', markersize=5)
        ax[idx, 3].axis('off')
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


def CalculateRMSE(realDir, fakeDir):
    real, fake = torch.empty(), torch.empty()

    for r in os.listdir(realDir):
        real.append(torch.load(r))

    for f in os.listdir(fakeDir):
        fake.append(torch.load(f))

    assert len(real) == len(fake)

    rmse = torch.empty()

    for x in range(len(real)):
        diff = fake[x] - real[x]
        rmse[x] = torch.sqrt(torch.mean(diff*diff))

    return torch.mean(rmse)


DetectEdges('outputWeights')

