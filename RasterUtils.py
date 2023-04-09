import torch
import numpy as np
from osgeo import gdal
from PIL import Image
import pds4_tools


def SaveDEM(filePath):
    """Save a tensor as a PDS4 file"""
    tensor = torch.load(filePath)[0].cpu()
    tensor = torch.squeeze(tensor)
    array = torch.Tensor.numpy(tensor)

    # save as image for reference
    Image.fromarray(np.uint8(array)).save('gdal_out_real.png', 'PNG')

    d_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("PDS4")
    raster = driver.Create('gdal_out_real.xml', array.shape[0], array.shape[1], 1, d_type)
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
    """Get the transformation and projection information from the raw DEM file"""
    driver = gdal.GetDriverByName('PDS4')
    driver.Register()
    file_name = 'Raw_DEMs/lrolrc_0042a/data/esm4/2019355/nac/m1331540878le.xml'
    data = gdal.Open(file_name)

    return data.GetGeoTransform(), data.GetProjectionRef()


def ViewPDS(filePath):
    """View a PDS file as an image"""
    struc = pds4_tools.read(filePath)
    struc.info()
    pds4_tools.view(filePath)

