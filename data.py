import os
from osgeo import gdal
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data_root, arg = True):
        self.root = data_root
        self.arg = arg
        self.rgb = []
        self.msi = []

        rgb_tr_path = f'{data_root}/Train_RGB/'
        msi_tr_path = f'{data_root}/Train_MSI/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as li:
            self.rgb_list = [line.replace('\n', '.jpg') for line in li]
            self.msi_list = [line.replace('jpg', 'tif') for line in self.rgb_list]
        self.rgb_list.sort()
        self.msi_list.sort()
        print(f'len(train_rgb) dataset:{len(self.rgb_list)-1}')
        print(f'len(train_multispectral) dataset:{len(self.msi_list)-1}')

        for i in range(len(self.rgb_list)):
            rgb_path = os.path.join(rgb_tr_path, self.rgb_list[i])
            rgb_data = cv2.imread(rgb_path) # uint8 (340, 340, 3)
            rgb_data = np.transpose(rgb_data, [2, 0, 1]) # uint8 (3, 340, 340)
            rgb_data = torch.tensor(rgb_data) # 转tensor
            self.rgb.append(rgb_data)
            print("\rRead [{}] processing [{}/{}]".format(self.rgb_list[i], i, len(self.rgb_list)-1), end="")  # RGB
        print()
        for i in range(len(self.msi_list)):
            msi_path = os.path.join(msi_tr_path, self.msi_list[i])
            msi_data = gdal.Open(msi_path).ReadAsArray()
            msi_data = torch.tensor(msi_data) # 转tensor
            self.msi.append(msi_data) # uint8 (4, 340, 340)
            print("\rRead [{}] processing [{}/{}]".format(self.msi_list[i], i, len(self.msi_list)-1), end="")  # MSI
        print()

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # tensor -> array
        img = img.numpy()

        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()

        # array -> tensor
        return torch.from_numpy(img.copy())

    def __getitem__(self, idx):

        rgb = self.rgb[idx]
        msi = self.msi[idx]


        random.seed(0)

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            rgb = self.arguement(rgb, rotTimes, vFlip, hFlip)
            msi = self.arguement(msi, rotTimes, vFlip, hFlip)

        # 查看转换后的图像
        # cv2.imshow("rgb", np.transpose(rgb.numpy(), [1, 2, 0]))
        # cv2.imshow("msi", np.transpose(msi.numpy(), [1, 2, 0]))
        # cv2.waitKey(0)
        return rgb, msi

    def __len__(self):
        return len(self.rgb_list)


class ValidDataset(Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.rgb = []
        self.msi = []

        rgb_tr_path = f'{data_root}/Val_RGB/'
        msi_tr_path = f'{data_root}/Val_MSI/'

        with open(f'{data_root}/split_txt/val_list.txt', 'r') as li:
            self.rgb_list = [line.replace('\n', '.jpg') for line in li]
            self.msi_list = [line.replace('jpg', 'tif') for line in self.rgb_list]
        self.rgb_list.sort()
        self.msi_list.sort()
        print(f'len(val_rgb) dataset:{len(self.rgb_list)-1}')
        print(f'len(val_multispectral) dataset:{len(self.msi_list)-1}')

        for i in range(len(self.rgb_list)):
            rgb_path = os.path.join(rgb_tr_path, self.rgb_list[i])
            rgb_data = cv2.imread(rgb_path) # uint8 (340, 340, 3)
            rgb_data = np.transpose(rgb_data, [2, 0, 1]) # uint8 (3, 340, 340)
            rgb_data = torch.tensor(rgb_data)
            self.rgb.append(rgb_data)
            print("\rRead [{}] processing [{}/{}]".format(self.rgb_list[i], i, len(self.rgb_list)-1), end="")  # RGB
        print()
        for i in range(len(self.msi_list)):
            msi_path = os.path.join(msi_tr_path, self.msi_list[i])
            msi_data = gdal.Open(msi_path).ReadAsArray()
            msi_data = torch.tensor(msi_data)

            self.msi.append(msi_data) # uint8 (4, 340, 340)
            print("\rRead [{}] processing [{}/{}]".format(self.msi_list[i], i, len(self.msi_list)-1), end="")  # MSI
        print()

    def __getitem__(self, idx):

        rgb = self.rgb[idx]
        msi = self.msi[idx]

        return rgb, msi

    def __len__(self):
        return len(self.rgb_list)



if __name__ == '__main__':
    t = TrainDataset('./dataset')
    v = ValidDataset('./dataset')
    # print(t[1])
    # print(v[1])