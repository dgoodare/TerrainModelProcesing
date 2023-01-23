from PIL import Image
from itertools import product
import numpy as np
import os
import csv
import time

"""
    Includes functions for applying masks to images and creating a lookup table that contains:
        - the original (ground truth) images
        - several versions of each image with different masks applied
        - the shape of the mask, for each masked image
"""


def sliceImage(filename, dir_in, dir_out, d):
    """Slices an input image into squares of size d"""
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    # create a grid with grid size d
    grid = product(range(0, h - h % d, d), range(0, w - w % d, d))

    # iterate through the grid saving each image to a file
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


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
    except OSError:
        print(f"{filename} could not be saved, or the file only contains partial data")


def saveMaskToFile(outDir, filename, content):
    """Saves a numpy array object to a npy file"""
    try:
        np.save(outDir + filename, content)
    except OSError:
        print(f"{filename} could not be saved, or the file only contains partial data")


def applyMasks(inputDir, imageOut, maskOut):
    """ Applies selection of masks to all images in a given image directory and saves the resulting and the
        corresponding mask to files in separate output folders.
        Returns a list, where each item is a row to be written to a .csv file
    """
    inputDir = inputDir
    imageOutDir = imageOut
    maskOutDir = maskOut
    counter = 0
    outputList = []

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

        # add masks to images and store mask info
        squareImage, squareMask = CreateSquareMask(imgArray, 8)
        stripImage, stripMask = CreateStripMask(imgArray, 2)

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

        # add file info to output list
        row = [img, squareImageFile, squareMaskFile, stripImageFile, stripMaskFile]
        outputList.append(row)

        # increment counter
        counter += 1

    print('Masks applied...')
    return outputList


def createLookUp():
    csvRows = applyMasks('slicedImages', 'maskedImages/', 'outputMasks/')
    # csv filename
    csvFile = 'lookUpTable.csv'
    # column names for csv file
    csvFields = ['Original Filename',
                 'Square image filename',
                 'Square mask filename',
                 'Strip image filename',
                 'Strip mask filename']

    try:
        # write to csv file
        with open(csvFile, 'w') as csvFile:
            csvWriter = csv.writer(csvFile, dialect='excel')
            csvWriter.writerow(csvFields)
            csvWriter.writerows(csvRows)

        print('Lookup table created...')
    except OSError:
        print("Failed to create lookUpTable.csv")


def main():
    startTime = time.time()
    inputDir = 'inputImages'
    outputDir = 'slicedImages'
    tileSize = 64

    # slice original images into smaller squares
    for img in os.listdir(inputDir):
        sliceImage(img, inputDir, outputDir, tileSize)
    print('Images sliced...')

    # apply masks and create lookup table
    createLookUp()
    print(f"Dataset created in {time.time()-startTime} seconds")


main()
