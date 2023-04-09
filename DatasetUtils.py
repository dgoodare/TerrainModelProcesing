import torch
import numpy as np
# from skimage.draw import disk, ellipse, polygon
import os
import csv
import time
# from osgeo import gdal
import sys
import pandas as pd

img_size = 64


def crop_to_size(arr, x):
    """Crops an array to a size that is divisible by x"""
    h, w = arr.shape
    h_r = h - (h % x)
    w_r = w - (w % x)
    return arr[:h_r, :w_r]


def slice_DEM(arr, size, in_file, outDir):
    """Slices an input array into smaller sub-arrays"""
    arr = crop_to_size(arr, size)
    h, w = arr.shape
    assert h % size == 0, f"{h} rows is not evenly divisible by {size}"
    assert w % size == 0, f"{w} cols is not evenly divisible by {size}"
    grid = arr.reshape(h // size, size, -1, size).swapaxes(1, 2).reshape(-1, size, size)

    idx = 1
    for x in grid:
        filename = outDir + '/' + in_file + '_' + str(idx) + '.pt'
        try:
            t = torch.tensor(x)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)

        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
        idx += 1

        # set the maximum number of slices to 20,000
        if idx > 7500:
            break


def CreateSquareMask(holeSize):
    """ Adds a square mask to an image
        Returns the resulting image and the corresponding mask as a tuple
    """
    # create a matrix with the same size as the input image and fill it with 1s
    mask = np.ones([img_size, img_size], dtype=int)
    # define the mask boundaries
    x2 = int(img_size / 2 + holeSize / 2)
    y1 = int(img_size / 2 - holeSize / 2)
    x1 = int(img_size / 2 - holeSize / 2)
    y2 = int(img_size / 2 + holeSize / 2)
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([holeSize, holeSize, 3], dtype=int)
    return mask


def CreateStripMask(holeWidth):
    """ Adds a horizontal strip mask to an image
        Returns the resulting image and the corresponding mask as a tuple
        """
    # create a matrix with the same size as the input image and fill it with 1s
    mask = np.ones([img_size, img_size], dtype=int)
    # define the mask boundaries
    y1 = int(img_size/2 - holeWidth/2)
    y2 = int(img_size/2 + holeWidth/2)
    x1 = int(0)
    x2 = int(img_size)
    # fill the mask area with 0s
    mask[x1:x2, y1:y2] = np.zeros([img_size, holeWidth], dtype=int)
    return mask


def CreateCircleMask(radius):
    """Adds a circular mask to an image"""
    # create numpy array to store the mask
    mask = np.ones([img_size, img_size], dtype=int)

    # define the centre of the circle to be the centre of the image
    c_x, c_y = int(img_size/2), int(img_size/2)
    r, c = disk((c_x, c_y), radius)
    mask[r, c] = 0

    return mask


def CreateEllipseMask(r_radius, c_radius):
    """Adds an elliptical mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)
    # set the centre of the ellipse to be the centre of the image
    c_x, c_y = int(img_size/2), int(img_size/2)
    r, c = ellipse(c_x, c_y, r_radius, c_radius)
    mask[r, c] = 0

    return mask


def CreatePolygonMask():
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [int(img_size/2), int(img_size/2)]  # centre of the image
    b = [int(img_size/2), 0]  # centre at the top edge
    c = [int(3*img_size/5), 0]  # 3/5 along the top edge
    # define row coordinates
    rows = np.array([a[1], b[1], c[1]])
    # define column coordinates
    cols = np.array([a[0], b[0], c[0]])
    # create polygon
    r, c = polygon(rows, cols)
    # fill mask
    mask[r, c] = 0

    return mask


def CreateTopLeftEdgeMask():
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [0, 0]  # top left corner of the image
    b = [int(2*img_size/10), 0]  # top edge
    c = [0, int(3*img_size/10)]  # left edge
    # define row coordinates
    rows = np.array([a[1], b[1], c[1]])
    # define column coordinates
    cols = np.array([a[0], b[0], c[0]])
    # create polygon
    r, c = polygon(rows, cols)
    # fill mask
    mask[r, c] = 0

    return mask


def CreateTopRightEdgeMask():
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)
    # define coordinates for the vertices of the polygon
    a = [img_size-1, 0]  # top left corner of the image
    b = [int(8*img_size/10), 0]  # top edge
    c = [img_size-1, int(1*img_size/10)]  # left edge
    # define row coordinates
    rows = np.array([a[1], b[1], c[1]])
    # define column coordinates
    cols = np.array([a[0], b[0], c[0]])
    # create polygon
    r, c = polygon(rows, cols)
    # fill mask
    mask[r, c] = 0

    return mask


def CreateTopLeftStripMask():
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)

    # define coordinates for the vertices of the polygon
    a = [0, int(img_size * (2.5/10))]
    b = [0, int(img_size * (3/10))]
    c = [int(img_size * (3/10)), 0]
    d = [int(img_size * (2.5/10)), 0]
    # define column coordinates
    cols = np.array([a[0], b[0], c[0], d[0]])
    # define row coordinates
    rows = np.array([a[1], b[1], c[1], d[1]])

    # create polygon
    r, c = polygon(rows, cols)

    # fill mask
    mask[r, c] = 0

    return mask


def CreateBottomRightStripMask():
    """Adds a polygon mask to an image"""
    # create numpy array
    mask = np.ones([img_size, img_size], dtype=int)

    # define coordinates for the vertices of the polygon
    a = [img_size-1, int(img_size * (2/10))]
    b = [img_size-1, int(img_size * (3/10))]
    c = [int(img_size * (7/10)), img_size-1]
    d = [int(img_size * (6.5/10)), img_size-1]
    # define column coordinates
    cols = np.array([a[0], b[0], c[0], d[0]])
    # define row coordinates
    rows = np.array([a[1], b[1], c[1], d[1]])

    # create polygon
    r, c = polygon(rows, cols)

    # fill mask
    mask[r, c] = 0

    return mask


def createMasks(shapeList, outDir):
    if shapeList[0]:
        mask = CreateTopLeftEdgeMask()
        filename = outDir + '/' + "tl_edge.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[1]:
        mask = CreateTopRightEdgeMask()
        filename = outDir + '/' + "tr_edge.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[2]:
        mask = CreateTopLeftStripMask()
        filename = outDir + '/' + "tl_strip.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[3]:
        mask = CreateBottomRightStripMask()
        filename = outDir + '/' + "br_edge.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[4]:
        mask = CreateSquareMask(int(img_size/4))
        filename = outDir + '/' + "sqr.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[5]:
        mask = CreateStripMask(int(img_size/8))
        filename = outDir + '/' + "c_strip.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[6]:
        mask = CreateCircleMask(int(img_size/4))
        filename = outDir + '/' + "circle.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[7]:
        mask = CreateEllipseMask(int(img_size/3), int(img_size/6))
        filename = outDir + '/' + "ellipse.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")
    if shapeList[8]:
        mask = CreatePolygonMask()
        filename = outDir + '/' + "polygon.pt"
        try:
            t = torch.from_numpy(mask)
            t = torch.unsqueeze(t, 0)
            torch.save(t, filename)
        except OSError:
            print(f"{filename} could not be saved, or the file only contains partial data")


def createRows(shapesList, inDir):
    outputList = []

    for file in os.listdir(inDir):
        if shapesList[0]:
            row = file, "tl_edge.pt"
            outputList.append(row)
        if shapesList[1]:
            row = file, "tr_edge.pt"
            outputList.append(row)
        if shapesList[2]:
            row = file, "tl_strip.pt"
            outputList.append(row)
        if shapesList[3]:
            row = file, "br_edge.pt"
            outputList.append(row)
        if shapesList[4]:
            row = file, "sqr.pt"
            outputList.append(row)
        if shapesList[5]:
            row = file, "c_strip.pt"
            outputList.append(row)
        if shapesList[6]:
            row = file, "circle.pt"
            outputList.append(row)
        if shapesList[7]:
            row = file, "ellipse.pt"
            outputList.append(row)
        if shapesList[8]:
            row = file, "polygon.pt"
            outputList.append(row)

    return outputList


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

    createMasks(shapes, 'outputMasks')
    print("masks created...")
    numShapes = 0
    for shape in shapes:
        if shape:
            numShapes += 1

    csvRows = createRows(shapes, 'outputSlices')
    print("rows created")
    # csv filename
    csvFile = 'LookUp/lookUpTable.csv'

    try:
        # write to csv file
        with open(csvFile, 'w', newline='') as csvFile:
            csvWriter = csv.writer(csvFile, dialect='excel')
            csvWriter.writerows(csvRows)

        print('Lookup table created...')
    except OSError:
        print("Failed to create lookUpTable.csv")

    return


def Create():
    """Given an PDS4 input, creates a dataset of slices from the input DEM"""
    startTime = time.time()
    driver = gdal.GetDriverByName('PDS4')
    driver.Register()

    file_name = 'Raw_DEMs/lrolrc_0042a/data/esm4/2019355/nac/m1331540878le.img'
    data = gdal.Open(file_name)

    if data is None:
        print('Unable to open file')
        sys.exit()

    print(f"Cols: {data.RasterXSize}, Rows: {data.RasterYSize}, bands: {data.RasterCount}")

    np_array = np.array(data.GetRasterBand(1).ReadAsArray(), dtype='f')

    # remove void data at the edges of the DEM
    h, w = np_array.shape
    trimmed = np_array[0:h, 50:(w-50)]

    slice_DEM(trimmed, img_size, 'm1331540878le', 'outputSlices')
    createLookUp()
    print(f"Dataset created in {time.time()-startTime:.4f} seconds")


def Clean(batchSize):
    """Ensures that the dataset is the correct size"""
    filePath = 'LookUp/lookUpTable.csv'
    lookUp = pd.read_csv(filePath)
    length = len(lookUp)
    excess = length % batchSize
    if excess > 0:
        diff = length - excess
        lookUp = lookUp.iloc[:diff]
        print(f"Dataset trimmed to fit with a batch size of {batchSize}")
        lookUp.to_csv(filePath, index=False)

