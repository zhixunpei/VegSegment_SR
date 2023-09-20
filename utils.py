import torch
import torch.nn as nn
import logging
import numpy as np
import os

from osgeo import gdal
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# 初始化log文件
def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


# 保存权重
def save_checkpoint(model_path, epoch, model, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

# 计算损失-平均绝对误差
class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error)   # .contiguous().view(-1)
        return mrae

# 计算损失-均方根误差
class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

# 计算损失-峰值信噪比
class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        mse = nn.MSELoss(reduction='none')
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# psnr
class T_Loss_PSNR(nn.Module):
    def __init__(self):
        super(T_Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake):
        im_true = (im_true.detach().cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8)  # 转numpy (b,c,h,w)
        im_fake = (im_fake.detach().cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8)  # 转numpy (b,c,h,w)

        # print(im_true.dtype, im_fake.shape[0])
        i_psnr = 0
        for i in range(im_true.shape[0]):
            i_t = im_true[i, ...]
            i_f = im_fake[i, ...]
            p = psnr(i_t, i_f)
            i_psnr += p

        m_psnr = i_psnr / im_true.shape[0]

        return m_psnr

# ssim
class T_Loss_SSIM(nn.Module):
    def __init__(self):
        super(T_Loss_SSIM, self).__init__()

    def forward(self, im_true, im_fake):
        im_true = (im_true.detach().cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8) # 转numpy (b,c,h,w)
        im_fake = (im_fake.detach().cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8) # 转numpy (b,c,h,w)

        # print(im_true.dtype, im_fake.shape[0])

        i_ssim = 0
        for i in range(im_true.shape[0]):
            i_t = im_true[i,...]
            i_f = im_fake[i,...]
            s = ssim(i_t, i_f, multichannel=True)
            i_ssim += s

        m_ssim = i_ssim / im_true.shape[0]

        return m_ssim


def outi(fakei, dir, name):
    fakei = fakei.detach().cpu().numpy()

    fakei = fakei[0, ...]

    fake_gre = fakei[0, ...]
    fake_red = fakei[1, ...]
    fake_reg = fakei[2, ...]
    fake_nir = fakei[3, ...]

    y = fake_gre.shape[0]
    x = fake_gre.shape[1]

    savepath = os.path.join(dir, f'{name}out.TIF') # 生成图信息
    result = gdal.GetDriverByName('GTiff').Create(savepath, xsize=x, ysize=y, bands=4, eType=gdal.GDT_Byte)
    result.GetRasterBand(1).WriteArray(fake_gre)
    result.GetRasterBand(2).WriteArray(fake_red)
    result.GetRasterBand(3).WriteArray(fake_reg)
    result.GetRasterBand(4).WriteArray(fake_nir)
    print("save: " + savepath)