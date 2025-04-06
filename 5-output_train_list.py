#!/.conda/envs/learn python
# -*- coding: utf-8 -*-

"""
样本制作第五步
输出模型训练的样本影像和标签列表
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
751984964@qq.com
"""

import fnmatch
import os

traintxt_path = r'E:\project_global_populus\MAE_test_250324\2-train_list\trainlist-250324.txt' #输出栅格样本列表文件路径
image_path = r'E:\project_global_populus\MAE_test_250324\2-enhance_img_224' #存储样本影像的文件夹路径
label_path = r'E:\project_global_populus\MAE_test_250324\2-enhance_label_224' #存储栅格标签的文件夹路径

img_data_tpye = '*.tif'
label_data_type = '*.tif'

img_list = fnmatch.filter(os.listdir(image_path), img_data_tpye) # 过滤文件
f = open(traintxt_path, 'wb')
for img_file in img_list:
    f.write((image_path + '/' + img_file[0:-4] + img_data_tpye[1:] + ' ').encode())
    f.write((label_path + '/' + img_file[0:-4] + label_data_type[1:] + '\n').encode())
f.close()

print('Finish')