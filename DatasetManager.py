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

img_size = 512  # the size of images in the training dataset


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
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
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
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
    # define the mask boundaries
    y1 = int(image.shape[1]/2 - holeWidth/2)
    y2 = int(image.shape[1]/2 + holeWidth/2)
    x1 = int(0)
    x2 = int(image.shape[0])
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([image.shape[0], holeWidth, image.shape[2]], dtype=int)
    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreateCircleMask(image, radius):
    """Adds a circular mask to an image"""
    # create numpy array to store the mask
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)

    # define the centre of the circle to be the centre of the image
    c_x, c_y = int(image.shape[0]/2), int(image.shape[1]/2)
    r, c = disk((c_x, c_y), radius)
    mask[r, c] = 0

    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreateEllipseMask(image, r_radius, c_radius):
    """Adds an elliptical mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
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
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
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


def CreateTopLeftEdgeMask(image):
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [0, 0]  # top left corner of the image
    b = [int(2*image.shape[0]/10), 0]  # top edge
    c = [0, int(3*image.shape[0]/10)]  # left edge
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


def CreateTopRightEdgeMask(image):
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [image.shape[1]-1, 0]  # top left corner of the image
    b = [int(8*image.shape[1]/10), 0]  # top edge
    c = [image.shape[1]-1, int(1*image.shape[0]/10)]  # left edge
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


def CreateTopLeftStripMask(image):
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)

    # define coordinates for the vertices of the polygon
    a = [0, int(image.shape[0] * (2.5/10))]
    b = [0, int(image.shape[0] * (3/10))]
    c = [int(image.shape[1] * (3/10)), 0]
    d = [int(image.shape[1] * (2.5/10)), 0]
    # define column coordinates
    cols = np.array([a[0], b[0], c[0], d[0]])
    # define row coordinates
    rows = np.array([a[1], b[1], c[1], d[1]])

    # create polygon
    r, c = polygon(rows, cols)

    # fill mask
    mask[r, c] = 0

    imageArray = np.multiply(image, mask)
    image = Image.fromarray(np.uint8(imageArray))
    return image, mask


def CreateBottomRightStripMask(image):
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([image.shape[0], image.shape[1], image.shape[2]], dtype=int)

    # define coordinates for the vertices of the polygon
    a = [image.shape[1]-1, int(image.shape[0] * (2/10))]
    b = [image.shape[1]-1, int(image.shape[0] * (3/10))]
    c = [int(image.shape[1] * (7/10)), image.shape[0]-1]
    d = [int(image.shape[1] * (6.5/10)), image.shape[0]-1]
    # define column coordinates
    cols = np.array([a[0], b[0], c[0], d[0]])
    # define row coordinates
    rows = np.array([a[1], b[1], c[1], d[1]])

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


def applyMasks(shapeList, inputDir, imageOut, maskOut):
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
        # noinspection PyTypeChecker
        imgArray = np.array(originalImg)

        row = [img]

        # Top left edge
        if shapeList[0]:
            img, mask = CreateTopLeftEdgeMask(imgArray)
            imgFile = "_tl_edge_img.png"
            maskFile = "_tl_edge_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Top right edge
        if shapeList[1]:
            img, mask = CreateTopRightEdgeMask(imgArray)
            imgFile = "_tr_edge_img.png"
            maskFile = "_tr_edge_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Top left strip
        if shapeList[2]:
            img, mask = CreateTopLeftStripMask(imgArray)
            imgFile = "_tl_strip_img.png"
            maskFile = "_tl_strip_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Bottom right strip
        if shapeList[3]:
            img, mask = CreateBottomRightStripMask(imgArray)
            imgFile = "_br_strip_img.png"
            maskFile = "_br_strip_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Square
        if shapeList[4]:
            img, mask = CreateSquareMask(imgArray, 16)
            imgFile = "_sqr_img.png"
            maskFile = "_sqr_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Centre strip
        if shapeList[5]:
            img, mask = CreateStripMask(imgArray, 16)
            imgFile = "_c_strip_img.png"
            maskFile = "_c_strip_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Circle
        if shapeList[6]:
            img, mask = CreateCircleMask(imgArray, 16)
            imgFile = "_circle_img.png"
            maskFile = "_circle_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Ellipse
        if shapeList[7]:
            img, mask = CreateEllipseMask(imgArray, 25, 16)
            imgFile = "_ellipse_img.png"
            maskFile = "_ellipse_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        # Polygon
        if shapeList[8]:
            img, mask = CreatePolygonMask(imgArray)
            imgFile = "_poly_img.png"
            maskFile = "_poly_mask.npy"
            saveImageToFile(imageOutDir, str(counter) + imgFile, img)
            saveMaskToFile(maskOutDir, str(counter) + maskFile, mask)
            row.append(str(counter) + imgFile)
            row.append(str(counter) + maskFile)

        outputList.append(row)

        # increment counter
        counter += 1

    print('Masks applied...')
    return outputList


def createCSVHeader(shapeList):
    fields = ['Original Image']

    if shapeList[0]:
        fields.append('tl_edge img')
        fields.append('tl_edge mask')

    if shapeList[1]:
        fields.append('tr_edge img')
        fields.append('tr_edge mask')

    if shapeList[2]:
        fields.append('tl_strip img')
        fields.append('tl_strip mask')

    if shapeList[3]:
        fields.append('br_strip img')
        fields.append('br_strip mask')

    if shapeList[4]:
        fields.append('sqr img')
        fields.append('sqr mask')

    if shapeList[5]:
        fields.append('c_strip img')
        fields.append('c_strip mask')

    if shapeList[6]:
        fields.append('circle img')
        fields.append('circle mask')

    if shapeList[7]:
        fields.append('ellipse img')
        fields.append('ellipse mask')

    if shapeList[8]:
        fields.append('poly img')
        fields.append('poly mask')

    print(f"Fields: {fields}")

    return fields


def createLookUp():
    # list of mask shapes to use
    shapes = [
        True,  # tl edge
        True,  # tr edge
        True,  # tl strip
        True,  # br strip
        False,  # square
        True,  # centre strip
        False,  # circle
        False,  # ellipse
        False  # polygon
    ]

    numShapes = 0
    for shape in shapes:
        if shape:
            numShapes += 1

    csvRows = applyMasks(shapes, 'slicedImages', 'maskedImages/', 'outputMasks/')
    # csv filename
    csvFile = 'LookUp/lookUpTable.csv'

    # column names for csv file
    csvFields = createCSVHeader(shapes)

    try:
        # write to csv file
        with open(csvFile, 'w') as csvFile:
            csvWriter = csv.writer(csvFile, dialect='excel')
            csvWriter.writerow(csvFields)
            csvWriter.writerows(csvRows)

        print('Lookup table created...')
    except OSError:
        print("Failed to create lookUpTable.csv")

    f = open("LookUp/lookUpTable.csv")
    numSamples = len(f.readlines()) * numShapes
    print(f"Number of samples: {numSamples}")

    return numSamples


def main():
    startTime = time.time()
    inputDir = 'inputImages'
    outputDir = 'slicedImages'
    tileSize = img_size

    # slice original images into smaller squares
    for img in os.listdir(inputDir):
        sliceImage(img, inputDir, outputDir, tileSize)
    print('Images sliced...')

    # apply masks and create lookup table
    num_samples = createLookUp()

    # convert to tensor files

    print(f"Dataset created in {time.time()-startTime} seconds")


def testMask():
    originalImg = Image.open('inputImages/image1.jpeg')
    # convert to numpy array
    imgArray = np.array(originalImg)
    image, mask = CreateTopLeftStripMask(imgArray)

    saveImageToFile('maskedImages/', "testImage.png", image)
    saveMaskToFile('outputMasks/', "testMask.npy", mask)


# main()
