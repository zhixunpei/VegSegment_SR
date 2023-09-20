import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from osgeo import gdal
import os

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


import cv2


def outi_pers(fakei, dir, name):
    fakei = fakei.detach().cpu().numpy()

    fakei = fakei[0, ...]

    fake_gre = fakei[0, ...]
    y = fake_gre.shape[0]
    x = fake_gre.shape[1]

    result = gdal.GetDriverByName('GTiff').Create(os.path.join(dir, f'out{name}_gre0.TIF'), xsize=x, ysize=y, bands=1, eType=gdal.GDT_Byte)
    result.GetRasterBand(1).WriteArray(fake_gre)


    fake_red = fakei[1, ...]
    result = gdal.GetDriverByName('GTiff').Create(os.path.join(dir, f'out{name}_red1.TIF'), xsize=x, ysize=y, bands=1, eType=gdal.GDT_Byte)
    result.GetRasterBand(1).WriteArray(fake_red)

    fake_reg = fakei[2, ...]
    result = gdal.GetDriverByName('GTiff').Create(os.path.join(dir, f'out{name}_reg2.TIF'), xsize=x, ysize=y, bands=1, eType=gdal.GDT_Byte)
    result.GetRasterBand(1).WriteArray(fake_reg)

    fake_nir = fakei[3, ...]
    result = gdal.GetDriverByName('GTiff').Create(os.path.join(dir, f'out{name}_nir3.TIF'), xsize=x, ysize=y, bands=1, eType=gdal.GDT_Byte)
    result.GetRasterBand(1).WriteArray(fake_nir)


def gen_seg():
    msi_data = gdal.Open("./pred/batch_pre_res/IMG_220611_073703_0038__01out.tif").ReadAsArray() # uint8
    msi_data = torch.tensor(msi_data)  # 转tensor

    msi_data = msi_data.int()
    print(msi_data.dtype)

    s1 = 0

    # 0:gre   1:red   2:reg   3:nir
    x1, x2, x3, x4 = 1, 0, -1, -1
    s1 = x1 * msi_data[0,:,:] + x2 * msi_data[1,:,:] + x3 * msi_data[2,:,:] + x4 * msi_data[3,:,:]


    im_gray1 = np.array(s1.numpy())
    # 利用图像像素均值二值化
    avg_gray = np.average(im_gray1)

    if avg_gray < 0 :
        avg_gray = avg_gray * 1.25
    else:
        avg_gray = avg_gray * 0.75

    print(avg_gray)

    im_gray2 = np.where(im_gray1 < avg_gray, 255, 0)


    data = np.array(im_gray2, dtype='uint8')
    plt.imshow(data)
    plt.show()
    # cv2.imwrite("./pred/IMG_220611_073743_0046__04seg.png", data)


def show_seg():
    # imgfile = './pred/avg_test1_12pth/test2.jpg'
    # pngfile = './pred/avg_test1_12pth/2+3.png'
    #
    # img = cv2.imread(imgfile, 1)
    # mask = cv2.imread(pngfile, 0)
    #
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    #
    # img = img[:, :, ::-1]
    # img[..., 2] = np.where(mask == 1, 255, img[..., 2])
    #
    # plt.imshow(img)
    # plt.savefig("./pred/avg_test1_12pth/result_2+3.png")
    # plt.show()
    image1 = Image.open("./pred/avg_out4/test.jpg")
    image2 = Image.open("./pred/avg_out4/0-2-3.png")

    plt.figure()

    plt.subplot(221)
    plt.imshow(image1)

    plt.subplot(222)
    plt.imshow(image2)

    plt.subplot(223)
    plt.imshow(image1)
    plt.imshow(image2, alpha=0.5)

    plt.savefig("./pred/avg_out4/3.png")
    plt.show()



def compute_miou(pred, target, nclass):
    mini = 1

    pred = np.array(pred)
    target = np.array(target)
    # 计算公共区域
    intersection = pred * (pred == target)

    # 直方图
    area_inter, _ = np.histogram(intersection, bins=2, range=(mini, nclass))
    area_pred, _ = np.histogram(pred, bins=2, range=(mini, nclass))
    area_target, _ = np.histogram(target, bins=2, range=(mini, nclass))
    area_union = area_pred + area_target - area_inter

    # 交集已经小于并集
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    rate = max(area_inter) / max(area_union)
    return rate





if __name__ == '__main__':
    # nclass = 1
    # # target
    # target = [[0,0,0],
    #           [0,1,1],
    #           [0,1,1]]
    #
    # # pred
    # pred = [[1,1,0],
    #         [1,1,0],
    #         [0,0,0]]
    #
    # # 计算miou
    # rate = compute_miou(pred, target, nclass)
    # print(rate)


    gen_seg()
    # show_seg()






# 0 1 2 3
# 01 02 03
# 12 13
# 23
# 012 013 023 123
# 0123

