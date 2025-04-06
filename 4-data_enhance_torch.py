#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作的第四步
样本增强（torch版本）
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

from PIL import Image
from torchvision import transforms
import numpy as np
import os
import fnmatch
import torch
from tqdm import tqdm

def color2gray(image_rgb, rgb_mapping):
    image_rgb_shape = np.shape(image_rgb)
    gray_img = np.zeros(shape=(image_rgb_shape[0], image_rgb_shape[1]), dtype=np.uint8)
    image_rgb = np.array(image_rgb)
    ndim = image_rgb.ndim

    for map_idx, rgb in enumerate(rgb_mapping):
        if ndim == 3:
            idx = np.where((image_rgb[..., 0] == rgb[0]) & (image_rgb[..., 1] == rgb[1]) & (image_rgb[..., 2] == rgb[2]))
        elif ndim == 2:
            idx = np.where(image_rgb == rgb)
        else:
            print('不支持的数据维度')
            exit(-1)

        gray_img[idx] = map_idx

    return gray_img

def label_colormap(n_label=5):
    if n_label == 5:  
        cmap = np.array(
            [
                (0, 0, 0), #背景值
                (255, 0, 0),
                (0, 255, 0),
                (0, 255, 255),
                (255, 255, 0),
                (0, 0, 255),
            ],
            dtype=np.uint8,
        )
        return cmap
    if n_label == 2:  
        cmap = np.array(
            [
                (0, 0, 0), #背景值
                (255, 255, 255),
            ],
            dtype=np.uint8,
        )
        return cmap
    if n_label ==8:
        cmap = np.array(
            [
                (0), #背景值
                (100),
                (200),
                (300),
                (400),
                (500),
                (600),
                (700),
                (800),
            ],
            dtype=np.int,
        )
        return cmap

if torch.cuda.is_available():
    print("gpu cuda is available!")
    torch.cuda.manual_seed(666)
else:
    print("cuda is not available! cpu is available!")
    torch.manual_seed(666)
np.random.seed(666)

images_path = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2\image-uint8-432' #原始影像路径 栅格
label_path = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2\label-raster' #标签影像路径 栅格
save_img_path = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2\image-uint8-432-256' #保存增强后影像路径
save_label_path = r'D:\BaiduNetdiskDownload\MHdataset\MHparcel\hetian-GF2\label-raster-256' #保存增强后标签路径

expandNum = 32 #每个样本的扩充数目
randomCorpSize = 256 #随机裁剪后的样本大小
randomColorChangeRange = 0 #随机色彩变换范围 0~1，越大变化越强 #仅针对3波段影像
ifGIDDataset = False
GIDdatasetClassNum = 8

if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
if not os.path.exists(save_label_path):
    os.mkdir(save_label_path)

image_list = fnmatch.filter(os.listdir(images_path), '*.tif')  # 文件类型过滤

for img_name in tqdm(image_list):

    img_full_path = os.path.join(images_path + '/' + img_name)
    label_full_path = os.path.join(label_path + '/' + img_name[0:-4] + '.tif')

    '''读取img和label数据'''
    sr_img = Image.open(img_full_path)
    sr_label = Image.open(label_full_path)

    for i in range(expandNum):
        
        save_img_full_path = save_img_path + '/' + img_name[0:-4] + '_' + str(i) + '.tif'
        save_label_full_path = save_label_path + '/' + img_name[0:-4] + '_' + str(i) + '.tif'

        p1 = np.random.choice([0,1])
        p2 = np.random.choice([0,1])
        p3 = np.random.choice([0,45])
        im_aug = transforms.Compose([transforms.RandomCrop(randomCorpSize),      
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                transforms.ColorJitter(brightness=randomColorChangeRange, contrast=randomColorChangeRange, saturation=randomColorChangeRange, hue=randomColorChangeRange)])

        label_aug = transforms.Compose([transforms.RandomCrop(randomCorpSize),      
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2)
                ]) #标签颜色不能变换

        torch.manual_seed(i+627)
        new_sr_img = im_aug(sr_img)

        

        torch.manual_seed(i+627)
        
        if ifGIDDataset == False:
            new_label_img = label_aug(sr_label).convert('L') #默认情况下标签转为灰度图
        else:
            new_label_img = label_aug(sr_label)
            cmap = label_colormap(n_label = GIDdatasetClassNum) #颜色图映射
            new_label_img = Image.fromarray(color2gray(new_label_img, cmap))

        new_sr_img.save(save_img_full_path)
        new_label_img.save(save_label_full_path)


        