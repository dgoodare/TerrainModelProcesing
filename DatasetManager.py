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
    """ Adds a square mask to an image
        Returns the resulting image and the corresponding mask as a tuple
    """
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
    """ Adds a horizontal strip mask to an image
        Returns the resulting image and the corresponding mask as a tuple
        """
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


def saveImageToFile(outDir, filename, content):
    """Saves a PIL image object to a JPEG file"""
    try:
        content.save(outDir + filename, 'JPEG')
        print(f"{filename} saved to output folder")
    except OSError:
        print(f"{filename} could not be saved, or the file only contains partial data")


def saveMaskToFile(outDir, filename, content):
    """Saves a numpy array object to a npy file"""
    try:
        np.save(outDir + filename, content)
        print(f"{filename} saved to output folder")
    except OSError:
        print(f"{filename} could not be saved, or the file only contains partial data")


def applyMasks(inputDir, imageOut, maskOut):
    """Applies selection of masks to all images in a given image directory and saves the resulting and the
        corresponding mask to files in separate output folders
    """
    inputDir = inputDir
    imageOutDir = imageOut
    maskOutDir = maskOut
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

        # add masks to images
        squareImage = CreateSquareMask(imgArray, 128)[0]
        stripImage = CreateStripMask(imgArray, 32)[0]
        # retrieve mask info
        squareMask = CreateSquareMask(imgArray, 128)[1]
        stripMask = CreateStripMask(imgArray, 32)[1]

        # create file names for images and masks and save to file
        # images
        squareImageFile = str(counter) + "_squareImage" + ".jpg"
        saveImageToFile(imageOutDir, squareImageFile, squareImage)

        stripImageFile = str(counter) + "_stripImage" + ".jpg"
        saveImageToFile(imageOutDir, stripImageFile, stripImage)

        # masks
        squareMaskFile = str(counter) + "_squareMask" + ".npy"
        saveMaskToFile(maskOutDir, squareMaskFile, squareMask)

        stripMaskFile = str(counter) + "_stripMask" + ".npy"
        saveMaskToFile(maskOutDir, stripMaskFile, stripMask)

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


applyMasks('inputImages', 'outputImages/', 'outputMasks/')
