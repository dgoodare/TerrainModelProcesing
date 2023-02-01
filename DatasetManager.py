from PIL import Image
from itertools import product
import numpy as np
from skimage.draw import disk, ellipse, polygon
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


def CreateCircleMask(image, radius):
    """Adds a circular mask to an image"""
    # create numpy array to store the mask
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)

    # define the centre of the circle to be the centre of the image
    c_x, c_y = int(image.shape[0]/2), int(image.shape[1]/2)
    r, c = disk((c_x, c_y), 10)
    mask[r, c] = 0

    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreateEllipseMask(image, r_radius, c_radius):
    """Adds an elliptical mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)
    # set the centre of the ellipse to be the centre of the image
    c_x, c_y = int(image.shape[0]/2), int(image.shape[1]/2)
    r, c = ellipse(c_x, c_y, r_radius, c_radius)
    mask[r, c] = 0

    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreatePolygonMask(image):
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], 3], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [int(image.shape[1]/2), int(image.shape[0]/2)]  # centre of the image
    b = [int(image.shape[0]/2), 0]  # centre at the top edge
    c = [int(3*image.shape[0]/5), 0]  # 3/5 along the top edge
    # define row coordinates
    rows = np.array([a[1], b[1], c[1]])
    # define column coordinates
    cols = np.array([a[0], b[0], c[0]])
    # create polygon
    r, c = polygon(rows, cols)
    # fill mask
    mask[r, c] = 0

    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def saveImageToFile(outDir, filename, content):
    """Saves a PIL image object to a JPEG file"""
    try:
        content.save(outDir + filename, 'PNG')
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
        ellipseImage, ellipseMask = CreateEllipseMask(imgArray, 40, 20)
        polyImage, polyMask = CreatePolygonMask(imgArray)

        # create file names for images and masks and save to file
        # images
        ellipseImageFile = str(counter) + "_ellipseImage" + ".png"
        saveImageToFile(imageOutDir, ellipseImageFile, ellipseImage)

        polyImageFile = str(counter) + "polyImage" + ".png"
        saveImageToFile(imageOutDir, polyImageFile, polyImage)

        # masks
        ellipseMaskFile = str(counter) + "_ellipseMask" + ".npy"
        saveMaskToFile(maskOutDir, ellipseMaskFile, ellipseMask)

        polyMaskFile = str(counter) + "_polyMask" + ".npy"
        saveMaskToFile(maskOutDir, polyMaskFile, polyMask)

        # add file info to output list
        row = [img, ellipseImageFile, ellipseMaskFile, polyImageFile, polyMaskFile]
        outputList.append(row)

        # increment counter
        counter += 1

    print('Masks applied...')
    return outputList


def createLookUp():
    csvRows = applyMasks('slicedImages', 'maskedImages/', 'outputMasks/')
    # csv filename
    csvFile = 'LookUp/lookUpTable.csv'
    # column names for csv file
    csvFields = ['Original image',
                 'Ellipse image',
                 'Ellipse mask',
                 'Polygon image',
                 'Polygon mask']

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
    tileSize = 256

    # slice original images into smaller squares
    for img in os.listdir(inputDir):
        sliceImage(img, inputDir, outputDir, tileSize)
    print('Images sliced...')

    # apply masks and create lookup table
    createLookUp()
    print(f"Dataset created in {time.time()-startTime} seconds")


main()

