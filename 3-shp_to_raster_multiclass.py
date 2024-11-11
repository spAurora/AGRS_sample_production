#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作第三步
矢量样本栅格化

针对多类别样本
矢量文件中不同类别要素的label字段赋值1~255
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

import os
import sys
import fnmatch
import numpy as np
import gdal
import ogr

image_path = r'D:\WeChat Files\wxid_qg2ddhelak9h22\FileStorage\File\2024-11\1-clip_img_Pam' #存储样本影像的文件夹
label_path = r'D:\WeChat Files\wxid_qg2ddhelak9h22\FileStorage\File\2024-11\1-clip_shp_Pam' #存储人工勾画矢量的文件夹
save_path = r'D:\WeChat Files\wxid_qg2ddhelak9h22\FileStorage\File\2024-11\1-label_raster_Pam' #输出的矢量转栅格样本文件夹

class_num = 4 #类别数 不包含背景0

if not os.path.exists(save_path):
    os.mkdir(save_path)

img_list = fnmatch.filter(os.listdir(image_path), '*.tif') # 过滤出所有tif文件

'''逐影像'''
for img_file in img_list:
    '''预处理'''
    data = []
    image_file = os.path.join(image_path + '/' + img_file)
    label_file = os.path.join(label_path + '/' + img_file[0:-4] + '_label.shp')
    outRaster_file = os.path.join(save_path + '/' + img_file[0:-4] + '.tif')
    if os.path.exists(label_file) == False: # shp文件不存在跳过
        print('File not exist: ' + img_file[0:-4] + '_label.shp')
        continue

    image = gdal.Open(image_file)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    geotransform = image.GetGeoTransform()             
    ref = image.GetProjection()
    x_res = image.RasterXSize
    y_res = image.RasterYSize
    vector = ogr.Open(label_file)
    if vector == None:
        print('读取shp_label文件失败')
    layer = vector.GetLayer() 

    '''保存栅格样本文件'''
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outRaster_file, image.RasterXSize, image.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(ref)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()

    gdal.RasterizeLayer(ds, [1], layer, options=["ATTRIBUTE=label"])

    image = None
    ds = None
    print(outRaster_file + ' success')

os.remove('temp.tif') # 清理tmp_tif文件
