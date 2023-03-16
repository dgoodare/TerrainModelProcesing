import torch
import numpy as np
from osgeo import gdal, osr
import rasterio
import sys
from PIL import Image


def GDAL_saveDEM(filePath):
    tensor = torch.load(filePath)
    tensor = torch.squeeze(tensor)
    array = torch.Tensor.numpy(tensor)

    # save as image for reference
    Image.fromarray(np.uint8(array)).save('gdal_out.png', 'PNG')

    d_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("PDS4")
    raster = driver.Create('gdal_out.img', array.shape[0], array.shape[1], 1, d_type)
    band = raster.GetRasterBand(1)
    band.WriteArray(array)

    # get transform and projection info
    transformRef, projectionRef = GetGeoRefData()
    print(transformRef)
    print(projectionRef)
    raster.SetGeoTransform(transformRef)
    raster.SetProjection(projectionRef)

    band.FlushCache()


def GetGeoRefData():
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    file_name = 'LROLRC_0042A/lrolrc_0042a/data/esm4/2019355/nac/m1331540878le.img'
    data = gdal.Open(file_name)

    return data.GetGeoTransform(), data.GetProjectionRef()


def RasterIO_saveDEM(filePath):
    tensor = torch.load(filePath)
    tensor = torch.squeeze(tensor)
    array = torch.Tensor.numpy(tensor)

    # save as image for reference
    Image.fromarray(np.uint8(array)).save('rasterio_out.png', 'PNG')

    dtype = rasterio.float32
    transform = rasterio.transform.from_origin(0, 0, 1, 1)

    raster = rasterio.open(
        'rasterio_out.tif', 'w', driver='GTiff',
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=dtype, crs='', transform=transform
    )
    raster.write(array, 1)
    raster.close()


GDAL_saveDEM('outputSlices/m1331540878le_1.pt')
