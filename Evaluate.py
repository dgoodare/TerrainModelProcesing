import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import filters, feature
from skimage.color import rgb2gray
import os
import torch


def DetectEdges(directory):
    """Highlights prominent edges in images in a given directory"""
    originalList = []
    sobelList = []
    prewittList = []
    cannyList = []
    num_images = 5
    count = 0

    for file in os.listdir(directory):
        tensor = torch.load(directory + "/" + file)
        tensor = torch.squeeze(tensor)
        img = torch.Tensor.numpy(tensor)

        sobel = filters.sobel(img)
        prewitt = filters.prewitt(img)
        canny = feature.canny(img)

        originalList.append(img)
        sobelList.append(sobel)
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


DetectEdges("outputSlices")

