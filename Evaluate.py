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
from torchmetrics.image.inception import InceptionScore
from LoadModel import Load, Generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DetectEdges(directory):
    """Highlights prominent edges in images in a given directory"""
    originalList = []
    prewittList = []
    cannyList = []
    harrisList = []
    num_images = 3
    count = 0
    # dirLength = len(os.listdir(directory))
    # rand_idx = random.randrange(dirLength - num_images)

    for file in os.listdir(directory):  # [rand_idx:rand_idx+num_images]:
        tensor = torch.load(directory + "/" + file).cpu()

        for t in tensor:
            img = torch.Tensor.numpy(t)
            # prewitt edge detection
            prewitt = filters.prewitt(img)
            # canny edge detection
            canny = feature.canny(img)
            # harris corner detection
            harris = feature.corner_harris(img, k=0.01)
            peaks = feature.corner_peaks(harris, min_distance=7)
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
        ax[idx, 0].imshow(originalList[idx], cmap='gray')
        ax[idx, 0].axis('off')
        ax[idx, 1].imshow(prewittList[idx])
        ax[idx, 1].axis('off')
        ax[idx, 2].imshow(cannyList[idx])
        ax[idx, 2].axis('off')
        ax[idx, 3].imshow(originalList[idx], interpolation='nearest')
        ax[idx, 3].plot(peaks[:, 1], peaks[:, 0], '+r', markersize=5)
        ax[idx, 3].axis('off')
    f.tight_layout()
    plt.show()


def CalculateFID(real, fake):
    fid = FrechetInceptionDistance(feature=64).to(device)
    realDir = real
    fakeDir = fake
    realTensors = FolderToTensor(realDir).to(device)
    fakeTensors = FolderToTensor(fakeDir).to(device)

    fid.update(realTensors, real=True)
    fid.update(fakeTensors, real=False)
    score = fid.compute()

    return score


def CalculateInception(folder):
    inception = InceptionScore(feature=64).to(device)
    imgs = FolderToTensor(folder).to(device)
    inception.update(imgs)
    score = inception.compute()
    return score


def FolderToTensor(folder):
    t_list = list()

    for file in os.listdir(folder):
        f = torch.load(folder + '/' + file).to(device)
        f = f.repeat(3, 1, 1)
        f = f.to(torch.uint8)
        t_list.append(f)

    outTensor = torch.stack(t_list, 0)

    return outTensor


def CalculateRMSE(r, f):
    real = torch.load(r)
    fake = torch.load(f)

    diff = fake - real
    rmse = torch.sqrt(torch.mean(diff*diff))

    return rmse


def histogram(filepath):
    t = torch.load(filepath).cpu()
    print(t.shape)
    fake = torch.Tensor.numpy(torch.squeeze(t))
    plt.hist(fake)
    plt.show()
