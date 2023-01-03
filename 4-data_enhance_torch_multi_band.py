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
from PIL import Image
from noise import pnoise2, snoise2
import math


def perlin_array(shape=(200, 200),
                 scale=100, octaves=6,
                 persistence=0.5,
                 lacunarity=2.0,
                 seed=None):
    '''柏林噪声的生成函数
    Args:
        shape: 生成形状大小
        scale: 值越大，生成的噪声越均匀（晶格越少,插值越多）
        octaves: 值越小越模糊，最小值>0
        persistence: 值越大越破碎，足够大的时候像随机噪声
        lacunaruty: 值越大细节越多，但大的背景不变
        seed: 随机种子

        scale: number that determines at what distance to view the noisemap.
        octaves: the number of levels of detail you want you perlin noise to have.
        lacunarity: number that determines how much detail is added or removed at each octave (adjusts frequency).
        persistence: number that determines how much each octave contributes to the overall shape (adjusts amplitude).

    Returns:
        arr: 归一化后的柏林噪声矩阵

    Raises:
        无
    '''

    if not seed:

        seed = np.random.randint(0, 100)
        #print("seed was {}".format(seed))

    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = pnoise2(i / scale,
                                j / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=1024,
                                repeaty=1024,
                                base=seed)
    max_arr = np.max(arr)
    min_arr = np.min(arr)

    def norm_me(x): return (x - min_arr) / (max_arr - min_arr)
    '''
    函数向量化
    如果一个python函数只能对单个元素进行某种处理操作，经过vectorize转化之后，能够实现对一个数组进行处理
    '''
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr


def AddHaze(img, perlin):
    '''影像加雾
    Args:
        img: 原始影像，值域[0,255]
        perlin: 柏林噪声矩阵, 值域[0,1]，尺寸与影像相同

    Returns:
        img_f: 加雾后的影像，值域[0,255]

    Raises:
        无
    '''
    img_f = img / 255.0
    (row, col) = img.shape

    # A = 0.37  # 定值亮度
    A = np.random.uniform(0.36, 0.38)
    beta = 0.06  # 雾的浓度

    norm = [0.9, 1]


    size = math.sqrt(max(row, col))  # 雾化尺寸
    #size = max(row, col)
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            # 以路径模拟云厚度
            # d = -0.04 * \
            #     math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            d = 5
            
            td = math.exp(-beta * (norm[0] + perlin[j][l]*(norm[1]-norm[0])) * d)
            img_f[j][l] = img_f[j][l] * td + A * (1 - td)
    return img_f * 255

def AddHaze2(img):
    '''简化版的影像加雾
    Args:
        img: 原始影像，值域[0,255]

    Returns:
        img_f: 加雾后的影像，值域[0,255]

    Raises:
        无
    '''
    img_h = img / 255.0  
    A = np.random.uniform(0.9, 1)
    t = np.random.uniform(0.3, 0.9)
    img_h = img_h*t + A*(1-t)

    return img_h*255

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


def write_img(out_path, im_data, mode=1, rotate=0, addHaze=False):
    """output img

    Args:
        out_path: Output path
        im_proj: Affine transformation parameters
        im_geotrans: spatial reference
        im_data: Output image data

    """
    # 生成随机种子
    seed = np.random.randint(0, 100)
    # identify data type
    if mode == 0:
        datatype = gdal.GDT_Byte
    else:
        datatype = gdal.GDT_Byte

    # calculate number of bands
    if len(im_data.shape) > 2:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # create new img
    driver = gdal.GetDriverByName("GTiff")
    new_dataset = driver.Create(
        out_path, im_width, im_height, im_bands, datatype)

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

        '''加云'''
        if addHaze == True:
            perlin_noise = perlin_array(shape=(im_height, im_width), scale=200,
                                            octaves=6, persistence=0.5, lacunarity=2.0, seed=seed)
            tmp = AddHaze(tmp, perlin_noise)

        new_dataset.GetRasterBand(i + 1).WriteArray(tmp)

    del new_dataset


images_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-clip_img'  # 原始影像路径 栅格
label_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-raster_label'  # 标签影像路径 栅格
save_img_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-enhance_img_addhaze_test'  # 保存增强后影像路径
save_label_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-enhance_label_addhaze_test'  # 保存增强后标签路径

expandNum = 8  # 每个样本的基础扩充数目，最终数目会在基础扩充数目上*6
randomCorpSize = 256  # 随机裁剪后的样本大小
img_edge_width = 512  # 输入影像的大小
add_haze_rate = 0.6  # 加雾的图像比例

max_thread = randomCorpSize / img_edge_width
print(max_thread)

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

        # p1 = np.random.choice([0, max_thread])  # 最大height比例
        # p2 = np.random.choice([0, max_thread])  # 最大width比例 巨大bug
        
        p1 = np.random.uniform(0, 1-max_thread)
        p2 = np.random.uniform(0, 1-max_thread)

        print(p1, p2)

        start_x = int(p1 * img_edge_width)
        start_y = int(p2 * img_edge_width)

        print('start x y: ', start_x, start_y)

        new_sr_img = sr_img[start_x:start_x + randomCorpSize,
                            start_y:start_y + randomCorpSize, :]
        new_label_img = sr_label[start_x:start_x +
                                 randomCorpSize, start_y:start_y + randomCorpSize]

        new_sr_img = new_sr_img.transpose(2, 0, 1)
        for j in range(6):
            save_img_full_path = save_img_path + '/' + \
                img_name[0:-4] + '_' + str(cnt) + '.tif'
            save_label_full_path = save_label_path + '/' + \
                img_name[0:-4] + '_' + str(cnt) + '.tif'
            cnt += 1
            l = np.random.uniform(0, 1)
            if l > add_haze_rate:
                addHaze = False
            else:
                addHaze = True
                print(i, cnt)
            write_img(save_img_full_path, new_sr_img,
                      mode=1, rotate=j, addHaze=addHaze)
            write_img(save_label_full_path, new_label_img,
                      mode=0, rotate=j, addHaze=False)
