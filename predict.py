import torch
import cv2
import numpy as np
from model import MST_Plus_Plus
from utils import outi
from exp import outi_pers

def create_model():
    model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4)
    return model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("--outf", type=str, default='./pred/', help='path MSI files')
    parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/net_12epoch.pth')
    parser.add_argument('--rgb_path', type=str, default='./hy_seg/1.jpg')

    args = parser.parse_args()

    return args


def main(args):

    model = create_model()
    pretrained_model_path = args.pretrained_model_path


    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['state_dict'])

    rgb_path = args.rgb_path
    rgb_data = cv2.imread(rgb_path)  # uint8 (340, 340, 3)
    rgb_data = np.transpose(rgb_data, [2, 0, 1])  # uint8 (3, 340, 340)
    rgb_data = np.float32(rgb_data)
    rgb_data = torch.tensor(rgb_data)  # 转tensor
    rgb_data = rgb_data.unsqueeze(0)
    print(rgb_data.dtype)


    MSI = model(rgb_data)

    print(MSI.shape)
    outi(MSI, args.outf, "hy_1")

    outi_pers(MSI, args.outf, "hy_1")



if __name__ == '__main__':
    args = parse_args()
    main(args)