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

os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data' #防止报error4错误

image_path = r'D:\img_8' #存储样本影像的文件夹
line_path = r'D:\img_8' #存储人工勾画矢量的文件夹
save_path = r'D:\img_8\out' #输出的矢量转栅格样本文件夹
num_classes = 2 #类别数 不包含背景0

data = []
img_list = fnmatch.filter(os.listdir(image_path), '*.tif') # 过滤出所有tif文件

'''逐影像'''
for img_file in img_list:
    '''预处理'''
    image_file = os.path.join(image_path + '/' + img_file)
    line_file = os.path.join(line_path + '/' + img_file[0:-4] + '_label.shp')
    outraster_file = os.path.join(save_path + '/' + img_file[0:-4] + '.tif')
    if os.path.exists(line_file) == False: # shp文件不存在跳过
        print('File not exist: ' + img_file[0:-4] + '_label.shp')
        continue

    image = gdal.Open(image_file)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    geotransform = image.GetGeoTransform()             
    ref = image.GetProjection()
    x_res = image.RasterXSize
    y_res = image.RasterYSize
    vector = ogr.Open(line_file)
    if vector == None:
        print('第二次shp文件失败')
    layer = vector.GetLayer() 

    '''逐类别'''
    for i in range(num_classes):
        focus_label_value = i+1
        tmp_shp_path = 'tmp_' + str(focus_label_value) + '.shp'
        newds = driver.CreateDataSource(tmp_shp_path) 
        newds.CopyLayer(layer, 'wHy')
        newds.Destroy()
        vector_copy = ogr.Open(tmp_shp_path, 1) # 读写方式打开
        layer_tmp = vector_copy.GetLayer()
        defn_tmp = layer_tmp.GetLayerDefn()
        fieldIndex = defn_tmp.GetFieldIndex('label')
        if fieldIndex < 0: # 若label字段不存在则报错
            print('label字段不存在')
            sys.exit(1)
        
        '''遍历要素并label不等于当前处理类别的要素'''
        feature = layer_tmp.GetNextFeature()
        fieldIndex = defn_tmp.GetFieldIndex('label')
        oField = defn_tmp.GetFieldDefn(fieldIndex)
        fieldName = oField.GetNameRef()
        while feature is not None:
            f_value = feature.GetField(fieldName) # 获取要素label字段的值
            if (f_value != focus_label_value):
                f_ID = feature.GetFID()
                layer_tmp.DeleteFeature(int(f_ID)) 
            feature = layer_tmp.GetNextFeature()
        
        '''栅格化'''
        targetDataset = gdal.GetDriverByName('GTiff').Create('temp.tif', x_res, y_res, 3, gdal.GDT_Byte)
        targetDataset.SetGeoTransform(image.GetGeoTransform())
        targetDataset.SetProjection(image.GetProjection())
        band = targetDataset.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.FlushCache()
        gdal.RasterizeLayer(targetDataset, [1], layer_tmp)
        targetDataset = None

        image_tmp = gdal.Open('temp.tif')
        data_tmp = image.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        data_tmp[np.where(data_tmp > 0)] = focus_label_value
        data.append(data_tmp)

        '''指针释放'''
        image_tmp = None
        vector_copy.Destroy()
        driver.DeleteDataSource(tmp_shp_path) # 清理tmp_shp文件

    vector.Destroy()

    '''叠置data处理'''
    data = np.array(data)
    data_shape = np.shape(data)

    data_out = np.zeros((int(data_shape[1]), int(data_shape[2])))
    for i in range(data_shape[1]):
        for j in range(data_shape[2]):
            data_out[i, j] = max(data[:, i, j])

    '''保存栅格样本文件'''
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(outraster_file, image.RasterXSize, image.RasterYSize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(ref)
    ds.GetRasterBand(1).WriteArray(data_out)
    image = None
    ds = None
    print(outraster_file + ' success')

os.remove('temp.tif') # 清理tmp_tif文件

