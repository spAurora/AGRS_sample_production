#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作的第四步
针对多波段影像的样本增强（大于4波段）
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""
import numpy as np
import os
import fnmatch
from tqdm import tqdm
import gdal
import sys
from skimage import transform


def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset == None:
        print('open sr_img false')
        sys.exit(1)
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_data = im_dataset.ReadAsArray(0, 0, im_width, im_height)
    del im_dataset

    return im_data

def write_img(out_path, im_data, mode = 1):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # identify data type 
    if mode == 0:
        datatype = gdal.GDT_Byte
    else:
        datatype = gdal.GDT_Float32

    # calculate number of bands
    if len(im_data.shape) > 2:  
        im_bands, im_height, im_width = im_data.shape
    else:  
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del new_dataset

images_path = r'G:\Huyang_test\1-clip_img' #原始影像路径 栅格
label_path = r'G:\Huyang_test\1-raster_label' #标签影像路径 栅格
save_img_path = r'G:\Huyang_test\2-enhance_img' #保存增强后影像路径
save_label_path = r'G:\Huyang_test\2-enhance_label' #保存增强后标签路径

expandNum = 6 # 每个样本的基础扩充数目，最终扩充数目为expandNum*4
randomCorpSize = 256 # 随机裁剪后的样本大小
img_edge_width = 512 # 输入影像的大小

max_thread = randomCorpSize/img_edge_width

image_list = fnmatch.filter(os.listdir(images_path), '*.tif')  # 过滤出tif文件

for img_name in tqdm(image_list):
    img_full_path = os.path.join(images_path + '/' + img_name)
    label_full_path = os.path.join(label_path + '/' + img_name[0:-4] + '.tif')

    '''读取img和label数据'''
    sr_img = read_img(img_full_path)
    sr_label = read_img(label_full_path)

    sr_img = sr_img.transpose(1, 2, 0)

    '''样本扩增'''
    cnt = 0
    for i in range(expandNum):

        p1 = np.random.choice([0, max_thread]) # 最大height比例
        p2 = np.random.choice([0, max_thread]) # 最大width比例

        start_x = int(p1*img_edge_width)
        start_y = int(p2*img_edge_width)

        new_sr_img = sr_img[start_x:start_x+randomCorpSize, start_y:start_y+randomCorpSize, :]
        new_label_img = sr_label[start_x:start_x+randomCorpSize, start_y:start_y+randomCorpSize]

        new_sr_img = new_sr_img.transpose(2, 0, 1)

        for i in range(4):
            new_sr_img =transform.rotate(new_sr_img, 90*i)
            new_label_img =transform.rotate(new_label_img, 90*i)

            save_img_full_path = save_img_path + '/' + img_name[0:-4] + '_' + str(cnt) + '.tif'
            save_label_full_path = save_label_path + '/' + img_name[0:-4] + '_' + str(cnt) + '.tif'
            cnt = cnt + 1

            write_img(save_img_full_path, new_sr_img)
            write_img(save_label_full_path, new_label_img, mode=0)




        