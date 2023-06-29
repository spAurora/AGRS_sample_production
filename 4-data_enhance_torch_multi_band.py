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
from scipy.signal import convolve2d

def AddHaze_ATSC(img, band_index):
    '''
       原始线性大气散射模型，该方法全部置conv_w=0
       k和d通过暴力搜索得到
    '''
    (row, col) = img.shape

    k = [0.448, 0.671, 0.605, 0.507, 0.603, 0.233, 0.265, 0.241]
    d = [45.9, 41, 46.1, 55.7, 53.6, 66.3, 64.8, 57.8]

    for j in range(row):
        for l in range(col):          
            img[j][l] = img[j][l] * k[band_index] + d[band_index]

    data_output = np.round(img).astype(np.uint8)

    return data_output

def perlin_array(shape=(256, 256),
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
        print("seed was {}".format(seed))

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

def AddHaze_ATSC_convw_perlin(img, band_index, apply_conv = True, apply_perlin = True, seed = None, dis_rate=6):
    '''影像加雾
       大气散射模型
       柏林噪声和遥感交叉辐射作为自由项
    '''
    (row, col) = img.shape

    k = [0.511, 0.526, 0.519, 0.514, 0.505, 0.346, 0.298, 0.276]
    d = [34.4, 37.9, 40.2, 42.4, 44.8, 53.2, 62.7, 54.5] 
    conv_w_list = [0.075, 0.015, 0, 0.038, 0, 0, 0.02, 0.019]

    convw_expand_rate = 2
    for i in range(len(conv_w_list)):
        conv_w_list[i] *= convw_expand_rate

    if not apply_conv: # 如果不开启交叉辐射则conv_w全部重置为0
        conv_w_list = [0, 0, 0, 0, 0, 0, 0, 0] 

    neighbour_weight = conv_w_list[band_index]

    kernel = np.array([[1/8, 1/5, 1/4, 1/5, 1/8], [1/5, 1/2, 1, 1/2, 1/5], [1/4, 1, 0, 1, 1/4], [1/5, 1/2, 1, 1/2, 1/5], [1/8, 1/5, 1/4, 1/5, 1/8]])
    kernel_sum = np.sum(kernel)
    kernel = kernel/kernel_sum
    kernel = kernel*neighbour_weight
    kernel[2][2] = 1-neighbour_weight

    img = convolve2d(img, kernel, mode='same')

    for j in range(row):
        for l in range(col):          
            img[j][l] = img[j][l] * k[band_index] + d[band_index]

    if apply_perlin: # 如果应用柏林噪声
        '''以下为柏林噪声相关参数'''
        ###########################
        scale = 60
        octaves = 8
        persistence = 0.5
        lacunarity = 2.0
        rate = dis_rate
        ###########################
        perlin = perlin_array(shape=(row, col), scale=scale,
                    octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed) # 定义柏林噪声矩阵

        perlin_dis = (perlin*rate)-(rate/2)
        perlin_dis = np.round(perlin_dis).astype('int8')

        img = img + perlin_dis # 将柏林噪声叠加到卷积计算后的图像上

    data_output = np.round(img).astype(np.uint8)

    return data_output

def read_img(sr_img):
    """read img

    Args:
        sr_img: The full path of the original image

    """
    im_dataset = gdal.Open(sr_img)
    if im_dataset is None:
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
    
    discrete_list = [4, 6, 8, 10, 12, 14] 

    seed = np.random.randint(1, 100)  # 柏林噪声种子
    seed_dis = np.random.randint(0, len(discrete_list)) # 离散程度种子

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
            tmp = AddHaze_ATSC_convw_perlin(tmp, i, apply_conv=True, apply_perlin=True, seed=seed, dis_rate=discrete_list[seed_dis]) # 在这里替换模拟云算法
            # tmp = AddHaze_ATSC(tmp, i)

        new_dataset.GetRasterBand(i + 1).WriteArray(tmp)

    del new_dataset


images_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-clip_img\1-clip_img_clear'  # 原始影像路径 栅格
label_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\1-raster_label\1-raster_label_clear'  # 标签影像路径 栅格
save_img_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-enhance_img\7-enhance_img_clear_mix_sim_haze_ATSC_LV2_convw_expand_rate_2_5_230515'  # 保存增强后影像路径
save_label_path = r'E:\xinjiang_huyang_hongliu\Huyang_test_0808\2-enhance_label\7-enhance_label_clear_mix_sim_haze_ATSC_LV2_convw_expand_rate_2_5_230515'  # 保存增强后标签路径

expandNum = 10  # 每个样本的基础扩充数目，最终数目会在基础扩充数目上*6
randomCorpSize = 256  # 随机裁剪后的样本大小
img_edge_width = 512  # 输入影像的大小
add_haze_rate = 0.2  # 加雾的图像比例

max_thread = randomCorpSize / img_edge_width

if not os.path.exists(save_img_path):
    os.mkdir(save_img_path)
if not os.path.exists(save_label_path):
    os.mkdir(save_label_path)

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
        
        p1 = np.random.uniform(0, 1-max_thread)
        p2 = np.random.uniform(0, 1-max_thread)

        start_x = int(p1 * img_edge_width)
        start_y = int(p2 * img_edge_width)

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
            l = np.random.uniform(0, 1) # 生成随机数
            if l > add_haze_rate: # 判断随机数是否大于一个比例
                addHaze = False
            else:
                addHaze = True # 添加模拟云
                print(i, cnt)
            write_img(save_img_full_path, new_sr_img,
                      mode=1, rotate=j, addHaze=addHaze)
            write_img(save_label_full_path, new_label_img,
                      mode=0, rotate=j, addHaze=False)
