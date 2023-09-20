import torch
import cv2
import os
import numpy as np
from model import MST_Plus_Plus
from other_model import AWAN, HRNET, HSCNN_Plus, MIRNet, HDNet, MPRNet
from utils import outi
from exp import outi_pers
from PIL import Image
import matplotlib.pyplot as plt
from osgeo import gdal
from utils import initialize_logger


def create_model(name):
    if name == 'MST++':
        model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4)
    elif name == 'HRnet':
        model = HRNET.SGN()  # 31689284 final // batch_size = 1
    elif name == 'HSCNN++':
        model = HSCNN_Plus.HSCNN_Plus()  # 299584 final // batch_size = 2
    elif name == 'HDnet':
        model = HDNet.HDNet()  # 2647552 final
    elif name == 'MPRnet':
        model = MPRNet.MPRNet()  # 60349 final
    else:
        print(f'Method {name} is not defined !!!!')

    return model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Predict")
    # parser.add_argument('--pretrained_model_path', type=str, default='./pred/single/HDnet/model/net_5epoch.pth')
    # parser.add_argument("--outf", type=str, default='./res/single/HDnet/005/', help='path MSI files')
    parser.add_argument('--rgb_dir', type=str, default='./pred/single/Val_RGB')

    # model_name : MST++  HRnet  HSCNN++  HDnet MPRnet
    parser.add_argument("--model_name", type=str, default='MST++', help='model name')

    args = parser.parse_args()

    return args


def gen_seg(tif_file, name, out_path):
    msi_data = gdal.Open(tif_file).ReadAsArray()
    msi_data = torch.tensor(msi_data)  # 转tensor

    # total_sum = torch.sum(msi_data[2,...])
    # print(total_sum)
    msi_data = msi_data.int()

    # ============fusion based on weight============
    s1 = 0
    # 0:gre   1:red   2:reg   3:nir
    x1, x2, x3, x4 = -1.0, 0.0, 1.0, 1.0
    s1 = x1 * msi_data[0, :, :] + x2 * msi_data[1, :, :] + x3 * msi_data[2, :, :] + x4 * msi_data[3, :, :]
    im_gray1 = np.array(s1.numpy())
    avg_gray = np.average(im_gray1)

    print(f'b{avg_gray}')

    # 聚合
    if avg_gray < 145:
        avg_gray = avg_gray * 1.15
    elif avg_gray > 150:
        avg_gray = avg_gray * 0.9

    print(avg_gray)
    im_gray2 = np.where(im_gray1 > avg_gray, 255, 0)

    # ============fusion based on VI============
    # vi = (msi_data[0, :, :] - msi_data[3, :, :])/(msi_data[0, :, :] + msi_data[3, :, :])
    # # vi = ((2 * msi_data[0, :, :]) - (msi_data[2, :, :] + msi_data[3, :, :]))/((2 * msi_data[0, :, :]) + (msi_data[2, :, :] + msi_data[3, :, :]))
    # # print(vi)
    # im_gray2 = np.where(vi < -0.2, 255, 0)

    # ============save result============
    data = np.array(im_gray2, dtype='uint8')
    cv2.imwrite(os.path.join(out_path, f'{name}.png'), data)
    print("save: " + os.path.join(out_path, f'{name}.png'))


def show_seg(rgb, seg, name, out_path):
    # ==============no.1==============
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

    # ==============no.2==============
    image1 = Image.open(rgb)
    image2 = Image.open(seg)

    plt.figure()

    plt.subplot(221)
    plt.imshow(image1)

    plt.subplot(222)
    plt.imshow(image2)

    plt.subplot(223)
    plt.imshow(image1)
    plt.imshow(image2, alpha=0.5)

    plt.savefig(os.path.join(out_path, f'{name}_c.png'))

    plt.close()
    # plt.show()

    # ==============no.3==============
    # image1 = Image.open(rgb)
    # image2 = Image.open(seg)
    #
    # image1 = image1.convert('RGBA')
    # image2 = image2.convert('RGBA')
    #
    # # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    # image = Image.blend(image1, image2, 0.3)
    # # image.save(os.path.join(out_path, f'{name}_c.png'))
    # image.show()


def main(args, pretrained_model_path, outf):
    model = create_model(args.model_name)

    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        # If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

    # # logging
    # log_dir = os.path.join(outf, f'{args.model_name}{pretrained_model_path}.log')
    # logger = initialize_logger(log_dir)

    rgb_dir = args.rgb_dir

    for rgb in os.listdir(rgb_dir):
        name = rgb.split(".")[0]

        rgb_data = cv2.imread(os.path.join(rgb_dir, rgb))  # uint8 (340, 340, 3)
        rgb_data = np.transpose(rgb_data, [2, 0, 1])  # uint8 (3, 340, 340)
        rgb_data = np.float32(rgb_data)
        rgb_data = torch.tensor(rgb_data)  # 转tensor
        rgb_data = rgb_data.unsqueeze(0)  # (1, 3, 340, 340)
        # print(rgb_data.dtype)

        MSI = model(rgb_data)

        outi(MSI, outf, rgb.split('.')[0])
        # outi_pers(MSI, outf, rgb.split('.')[0])

        gen_seg(os.path.join(outf, f'{name}out.TIF'), name, outf)

        show_seg(os.path.join(rgb_dir, rgb), os.path.join(outf, f'{name}.png'), name, outf)


if __name__ == '__main__':
    args = parse_args()

    for epoch in range(0, 100, 5):
        pdm_path = f'./pred/single/{args.model_name}/model/net_{epoch}epoch.pth'
        outf = f'./res/single/{args.model_name}/{epoch}/'

        if not os.path.exists(outf):
            os.makedirs(outf)

        main(args, pdm_path, outf)

    for epoch in range(100, 300, 20):
        pdm_path = f'./pred/single/{args.model_name}/model/net_{epoch}epoch.pth'
        outf = f'./res/single/{args.model_name}/{epoch}/'

        if not os.path.exists(outf):
            os.makedirs(outf)

        main(args, pdm_path, outf)