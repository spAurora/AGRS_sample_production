#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作的第一步
根据点shp文件，以各个点为中心，将周围的小块影像裁剪出来
*必须保证原始影像和点shp文件的投影一致
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

import gdal
import ogr
import fnmatch
import os
import sys
import numpy as np


def write_img(out_path, im_proj, im_geotrans, im_data):
    """输出影像

    目前仅支持tif格式

    Args:
        out_path: 输出路径
        im_proj: 输出图像的仿射矩阵
        im_geotrans 输出图像的空间参考
        im_data 输出图像的数据，以np.array格式存储

    Returns:
        none
    """
    # 判断数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 计算波段数
    if len(im_data.shape) > 2:  # 多波段
        im_bands, im_height, im_width = im_data.shape
    else:  # 单波段
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建新影像
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)
    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset


def clip(out_tif_name, sr_img, point_shp, cut_cnt):
    """裁剪主函数

    根据点shp从原始影像中裁剪出小块影像用来作样本

    Args:
        out_tif_name: 输出小tif影像的不完整路径 后面的_n.tif在该函数中补充
        sr_img: 原始影像的完整路径
        point_shp 点shp的完整路径

    Returns:
        none
    """
    # 读取原始影像
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)  # 0为正常，1~127为异常
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize

    # 读取样本点矢量文件
    shp_dataset = ogr.Open(point_shp)
    if shp_dataset == None:
        print('open shapefile false')
        sys.exit(1)
    layer = shp_dataset.GetLayer()
    point_proj = layer.GetSpatialRef()

    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geoX = geom.GetX()
        geoY = geom.GetY()
        g0 = float(im_geotrans[0])
        g1 = float(im_geotrans[1])
        g2 = float(im_geotrans[2])
        g3 = float(im_geotrans[3])
        g4 = float(im_geotrans[4])
        g5 = float(im_geotrans[5])

        x = (geoX*g5 - g0*g5 - geoX*g2 + g3*g2)/(g1*g5 - g4*g2)
        y = (geoY - g3 - geoX*g4)/ g5

        x, y = int(x), int(y)

        a1 = x - adatasize
        a2 = y - adatasize
        a3 = x + adatasize
        a4 = y + adatasize
        if a1 > 0 and a2 > 0 and a3 > 0 and a4 > 0 and a3 < im_width and a4 < im_height:
            cut_cnt = cut_cnt + 1
            geoX2 = g0 + g1 * a1 + g2 * a2
            geoY2 = g3 + g4 * a1 + g5 * a2
            im_data = im_dataset.ReadAsArray(a1, a2, datasize, datasize)
            im_geotrans_list = list(im_geotrans)
            im_geotrans_list[0] = geoX2
            im_geotrans_list[3] = geoY2
            strname = out_tif_name + '_' + str(cut_cnt) + '.tif'
            write_img(strname, im_proj, im_geotrans_list, im_data)
        feature.Destroy()
        feature = layer.GetNextFeature()
        

    shp_dataset.Destroy()
    return cut_cnt


# 防止GDAL报ERROR4错误 gdal_gata文件夹需要相应配置
os.environ['GDAL_DATA'] = r'C:\Users\75198\.conda\envs\learn\Lib\site-packages\GDAL-2.4.1-py3.6-win-amd64.egg-info\gata-data'

sr_image_path = r"E:\xinjiang_huyang_hongliu\WV_GF_Tarim\WV2_dealed\Talimu_dealed" #原始影像
point_shp = r"E:\xinjiang_huyang_hongliu\Huyang_test_0808\0-other_data\haze1_points_for_show.shp" #中心点point文件
out_path = r"C:\Users\75198\OneDrive\论文\SCI-3-3 Remote sensing data augmentation\图片\7-预测结果展示图\待预测原始影像\8波段" #输出目标文件夹
datasize = 256 #输出的影像大小（像素）
img_type = '*.dat' #原始影像类型 不可漏*.
output_prefix = 'haze1_for_predict_show' #输出小块影像文件名的前缀

if not os.path.exists(out_path):
    os.mkdir(out_path)

# 过滤出原始影像
sr_img_list = fnmatch.filter(os.listdir(sr_image_path), img_type)  # tif还是TIF注意区别

adatasize = int(datasize / 2)
print(sr_img_list)

cnt = 0
cut_cnt = 0
for sr_img in sr_img_list:
    cnt = cnt+1
    shp_name, extension = os.path.splitext(sr_img)
    sr_img = sr_image_path + '/' + shp_name + img_type[1:]
    out_tif_name = out_path + '/' + output_prefix #改输出编号 
    print('start clip image', cnt)
    cut_cnt = clip(out_tif_name, sr_img, point_shp, cut_cnt)
    print('clip image', cnt,'done')
print('Finish!')
