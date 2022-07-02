#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
输入样本栅格影像文件夹，批量生成对应的矢量文件并自动添加label字段，为勾样本做准备
地块样本制作的第二步
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
from numpy.core.fromnumeric import shape
import ogr
from osgeo.gdal import Dataset, Driver
import osr

os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO") #路径支持中文
gdal.SetConfigOption("SHAPE_ENCODING","") #属性表支持中文

image_path = r'E:\xinjiang\water\1-clip_srimg' #存储样本影像的文件夹
save_path = r'E:\xinjiang\water\0-otherdata\teach' #输出的矢量文件夹

ogr.RegisterAll()# 注册所有的驱动

image_list = fnmatch.filter(os.listdir(image_path), '*.tif')  # 过滤出tif文件

for img in image_list:
    full_img_path = image_path + '/' + img
    full_outshp_name = save_path + '/' + img[:-4] + '_label.shp'

    dataset = gdal.Open(full_img_path)
    if dataset == None:
        print("open img false")
        sys.exit(1)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_dataset = driver.CreateDataSource(full_outshp_name)
    if shp_dataset == None:
        print('创建shp文件失败')

    spatial_ref = osr.SpatialReference(wkt=dataset.GetProjection())
    oLayer = shp_dataset.CreateLayer("polygon", spatial_ref, ogr.wkbPolygon)  # 保证图层名与属性一致
    #oLayer = shp_dataset.CreateLayer("line", spatial_ref, ogr.wkbLineString)  # 保证图层名与属性一致
    oFieldID = ogr.FieldDefn("label", ogr.OFSTInt16)
    oLayer.CreateField(oFieldID, 1)

    shp_dataset.Destroy()