#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作第三步
矢量样本栅格化
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

import os
import sys
import fnmatch
import numpy as np
import shutil
import gdal
import ogr

os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

image_path = r'E:\manas_class\project_manas\glacier\1-clip_img' #存储样本影像的文件夹
line_path = r'E:\manas_class\project_manas\glacier\1-artificial_shp' #存储人工勾画矢量的文件夹
save_path = r'E:\manas_class\project_manas\glacier\1-raster_label' #输出的矢量转栅格样本文件夹
background_value = 0 #栅格化后背景值
foreground_value = 255 #栅格化后前景值

img_list = fnmatch.filter(os.listdir(image_path), '*.tif') # 过滤出所有tif文件

for img_file in img_list:
    '''预处理'''
    image_file = os.path.join(image_path + '/' + img_file)
    line_file = os.path.join(line_path + '/' + img_file[0:-4] + '_label.shp')
    outraster_file = os.path.join(save_path + '/' + img_file[0:-4] + '.tif')
    if os.path.exists(line_file) == False: #shp文件不存在跳过
        print('File not exist: ' + img_file[0:-4] + '_label.shp')
        continue
    
    '''自动添加label字段'''
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()
    ds = ogr.Open(line_file, 1)
    if ds == None:
        print('打开shp文件失败')
    layer = ds.GetLayerByIndex(0)
    defn = layer.GetLayerDefn()
    fieldIndex = defn.GetFieldIndex('label')
    if fieldIndex < 0: # 若字段不存在则创建字段
        fieldDefn = ogr.FieldDefn('label', ogr.OFTInteger)
        fieldDefn.SetPrecision(0)
        layer.CreateField(fieldDefn, 1)
    fieldIndex2 = defn.GetFieldIndex('label')
    if fieldIndex2 < 0:
        print("字段创建失败")

    feature = layer.GetNextFeature()
    index = defn.GetFieldIndex('label')
    oField = defn.GetFieldDefn(index)
    fieldName = oField.GetNameRef()
    while feature is not None:
        feature.SetField2(fieldName, 1) # 设置label字段的值
        layer.SetFeature(feature)
        feature = layer.GetNextFeature()
    ds.Destroy()


    '''勾画的线shp样本栅格化'''
    image = gdal.Open(image_file)
    geotransform = image.GetGeoTransform()             
    ref = image.GetProjection()
    x_res = image.RasterXSize
    y_res = image.RasterYSize
    vector = ogr.Open(line_file)
    if vector == None:
        print('第二次shp文件失败')
    layer = vector.GetLayer()
    targetDataset = gdal.GetDriverByName('GTiff').Create('temp.tif', x_res, y_res, 3, gdal.GDT_Byte)
    targetDataset.SetGeoTransform(image.GetGeoTransform())
    targetDataset.SetProjection(image.GetProjection())
    band = targetDataset.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(targetDataset, [1, 2, 3], layer, )
    targetDataset = None

    '''标签二值化 for 二分类'''
    image = gdal.Open('temp.tif')
    data = image.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    data[np.where(data==0)] = background_value # 样本背景值
    data[np.where(data>0)] = foreground_value # 样本前景值
    

    '''保存栅格样本文件'''
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outraster_file, image.RasterXSize, image.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(ref)
    ds.GetRasterBand(1).WriteArray(data)
    image = None
    ds = None
    print(outraster_file + ' success')

os.remove('temp.tif') # 清理缓存文件

