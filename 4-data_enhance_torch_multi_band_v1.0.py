#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作的第四步
针对多波段影像的样本增强（大于4波段）
实现了对多波段影像的色彩变换
~~~~~~~~~~~~~~~~
code by WHR
"""
import numpy as np
import os
import fnmatch
from tqdm import tqdm
import gdal
import sys
from skimage import transform
from PIL import Image
from numpy import linalg
import random

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

    return im_data,im_geotrans,im_proj

def write_img(im_geotrans,im_proj,out_path, im_data, mode = 1, rotate = 0):
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
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)
    

    for i in range(im_bands):
        if mode == 0:
            tmp = im_data
        elif mode == 1:
            tmp = im_data[i]
        else:
            print('mode should 0 or 1!')
        im = Image.fromarray(tmp)

        if rotate == 0:
            out = im
        elif rotate == 1:
            out = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif rotate == 2:
            out = im.transpose(Image.FLIP_TOP_BOTTOM)
        elif rotate == 3:
            out = im.transpose(Image.ROTATE_90)
        elif rotate == 4:
            out = im.transpose(Image.ROTATE_180)
        elif rotate == 5:
            out = im.transpose(Image.ROTATE_270)

        tmp = np.array(out)

        new_dataset.GetRasterBand(i + 1).WriteArray(tmp)

    del new_dataset

images_path = r'D:\森林草地样本\森林样本\test\1-clip_img' #原始影像路径 栅格
label_path = r'D:\森林草地样本\森林样本\test\2-raster_label' #标签影像路径 栅格
save_img_path = r'D:\森林草地样本\森林样本\test\2-enhance_img' #保存增强后影像路径
save_label_path = r'D:\森林草地样本\森林样本\test\2-enhance_label' #保存增强后标签路径

save_clip_path = r'D:\森林草地样本\森林样本\test\2-enhance_no'#保存仅做裁剪和旋转的影像用于对比

expandNum = 3 # 每个样本的基础扩充数目
randomCorpSize = 256 # 随机裁剪后的样本大小
img_edge_width = 1500 # 输入影像的大小

max_thread = randomCorpSize/img_edge_width

image_list = fnmatch.filter(os.listdir(images_path), '*.tif')  # 过滤出tif文件

for img_name in tqdm(image_list):
    img_full_path = os.path.join(images_path + '/' + img_name)
    label_full_path = os.path.join(label_path + '/' + img_name[0:-4] + '.tif')

    '''读取img和label数据'''
    sr_img,im_geotrans,im_proj = read_img(img_full_path)
    sr_label,_,_ = read_img(label_full_path)
    bandnum = sr_img.shape[0]

    sr_img = sr_img.transpose(1, 2, 0)

    '''样本扩增'''
    cnt = 0
    for i in range(expandNum):

        #先做随机裁剪
        p1 = np.random.choice([0, max_thread]) # 最大height比例
        p2 = np.random.choice([0, max_thread]) # 最大width比例

        start_x = int(p1*img_edge_width)
        start_y = int(p2*img_edge_width)

        sr_img_clip = sr_img[start_x:start_x+randomCorpSize, start_y:start_y+randomCorpSize, :]
        label_img_clip = sr_label[start_x:start_x+randomCorpSize, start_y:start_y+randomCorpSize]
        
        #再做随机色彩增强
        #数据标准化
        sr_img_tmp = sr_img_clip / 255.0
        mean = sr_img_tmp.mean(axis = 0).mean(axis = 0)
        std = sr_img_tmp.reshape((-1, 4)).std()
        sr_img_tmp = (sr_img_tmp - mean) / (std)
        sr_img_tmp = sr_img_tmp.reshape((-1, bandnum))

        cov = np.cov(sr_img_tmp, rowvar = False)# 求矩阵的协方差矩阵
        eigValue, eigVector = linalg.eig(cov)# 求协方差矩阵的特征值和向量

        rand = np.array([random.normalvariate(0, 0.05) for i in range(bandnum)])#设置抖动系数
        jitter = np.dot(eigVector, eigValue * rand)#计算每个波段的抖动值
        jitter = (jitter * 255).astype(np.int32)[np.newaxis, np.newaxis, :]#扩充维度
        sr_img_enhance = np.clip(sr_img_clip + jitter, 0, 255)#广播相加；np.clip防止加入抖动后越界[0,255]

        sr_img_enhance = sr_img_enhance.transpose(2, 0, 1)
        sr_img_clip = sr_img_clip.transpose(2, 0, 1)

        #最后做随机旋转
        save_img_full_path = save_img_path + '/' + img_name[0:-4] + '_' + str(cnt) + '.tif'
        save_label_full_path = save_label_path + '/' + img_name[0:-4] + '_' + str(cnt) + '.tif'
        save_clip_full_path = save_clip_path + '/' + img_name[0:-4] + '_' + str(cnt) + '.tif'
        cnt += 1
        rotate_mode = np.random.choice(6)
        
        write_img(im_geotrans,im_proj,save_img_full_path, sr_img_enhance, mode=1, rotate=rotate_mode)
        write_img(im_geotrans,im_proj,save_label_full_path, label_img_clip, mode=0, rotate=rotate_mode)
        write_img(im_geotrans,im_proj,save_clip_full_path, sr_img_clip, mode=1, rotate=rotate_mode)




        