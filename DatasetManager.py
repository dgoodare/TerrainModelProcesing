from PIL import Image
import numpy as np
import os

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
    return image


def CreateStripMask(image, holeWidth):
    # create a matrix with the same size as the input image and fill it with 1s
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)
    print(f"mask shape: {mask.shape}")
    # define the mask boundaries
    x1 = int(image.shape[0]/2 - holeWidth/2)
    x2 = int(image.shape[0]/2 + holeWidth/2)
    y1 = int(0)
    y2 = int(image.shape[1])
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([holeWidth, image.shape[1], 3], dtype=int)
    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image


def applyMasks(inputDir):
    inputDir = inputDir
    counter = 1

    for img in os.listdir(inputDir):
        print(type(img))
        originalImg = Image.open(inputDir + '/' + img)
        # convert to numpy array
        imgArray = np.array(originalImg)
        # add masks
        squareMask = CreateSquareMask(imgArray, 128)
        stripMask = CreateStripMask(imgArray, 32)
        # create file names
        squareFile = "squareMask" + str(counter) + ".jpg"
        stripFile = "stripMask" + str(counter) + ".jpg"
        # save in output folder
        squareMask.save('outputImages/' + squareFile, 'JPEG')
        stripMask.save('outputImages/' + stripFile, 'JPEG')
        # increment counter
        counter += 1


