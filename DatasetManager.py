from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

"""
    Includes functions for applying masks to images and creating a lookup table that contains:
        - the original (ground truth) images
        - several versions of each image with different masks applied
        - the shape of the mask, for each masked image
        
    It also contains a modified version of the PyTorch Dataset class 
"""


def CreateSquareMask(image, holeSize):
    # create a matrix with the same size as the input image and fill it with 1s
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)
    # define the mask boundaries
    x2 = int(image.shape[0] / 2 + holeSize / 2)
    y1 = int(image.shape[1] / 2 - holeSize / 2)
    x1 = int(image.shape[0] / 2 - holeSize / 2)
    y2 = int(image.shape[1] / 2 + holeSize / 2)
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([holeSize, holeSize, 3], dtype=int)
    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreateStripMask(image, holeWidth):
    # create a matrix with the same size as the input image and fill it with 1s
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)
    # define the mask boundaries
    y1 = int(image.shape[1]/2 - holeWidth/2)
    y2 = int(image.shape[1]/2 + holeWidth/2)
    x1 = int(0)
    x2 = int(image.shape[0])
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([image.shape[0], holeWidth, 3], dtype=int)
    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def applyMasks(inputDir):
    inputDir = inputDir
    counter = 0

    # open input directory and iterate through the images
    for img in os.listdir(inputDir):
        originalImg = Image.open(inputDir + '/' + img)

        # check if the file opened properly
        if originalImg is None:
            print(f"unable to open {originalImg}")
            print("cancelling mask application process...")
            break

        # convert to numpy array
        imgArray = np.array(originalImg)
        # add masks
        squareMask = CreateSquareMask(imgArray, 128)[0]
        stripMask = CreateStripMask(imgArray, 32)[0]

        # create file names
        squareFile = str(counter) + "squareMask" + ".jpg"
        stripFile = str(counter) + "stripMask" + ".jpg"

        # save in output folder
        try:
            squareMask.save('outputImages/' + squareFile, 'JPEG')
            print(f"{squareFile} saved to output folder")
        except OSError:
            print(f"{squareFile} could not be saved, or the file only contains partial data")

        try:
            stripMask.save('outputImages/' + stripFile, 'JPEG')
            print(f"{stripFile} saved to output folder")
        except OSError:
            print(f"{stripFile} could not be saved, or the file only contains partial data")

        # increment counter
        counter += 1


class DEMDataset(Dataset):
    """A modified version of the PyTorch Dataset class"""
    def __init__(self):
        return

    def __len__(self):
        return

    def __getitem__(self, item):
        return


applyMasks('inputImages')
