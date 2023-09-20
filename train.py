from data import TrainDataset, ValidDataset
from model import MST_Plus_Plus
from utils import Loss_MRAE, Loss_RMSE, Loss_PSNR, AverageMeter, initialize_logger, time2file_name, save_checkpoint, T_Loss_PSNR, T_Loss_SSIM, outi
from other_model import AWAN, HRNET, HSCNN_Plus, MIRNet, HDNet, MPRNet
from other_model import MST_o, MPRNet_o, HSCNN_Plus_o, HDNet_o, HRNET_o

import torch
import torch.backends.cudnn as cudnn
import datetime
import os


def create_model(name):

    if name == 'mst++':
        model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4)
    elif name == 'hrnet':
        model = HRNET.SGN() # 31689284 final // batch_size = 1
    elif name == 'hscnn_plus':
        model = HSCNN_Plus.HSCNN_Plus() # 299584 final // batch_size = 2
    elif name == 'hdnet':
        model = HDNet.HDNet() # 2647552 final
    elif name == 'mprnet':
        model = MPRNet.MPRNet() # 60349 final

    elif name == 'mprnet_o':
        model = MPRNet_o.MPRNet() # 60349 final
    elif name == 'hscnn_plus_o':
        model = HSCNN_Plus_o.HSCNN_Plus() # 60349 final
    elif name == 'hdnet_o':
        model = HDNet_o.HDNet() # 60349 final
    elif name == 'hrnet_o':
        model = HRNET_o.SGN() # 60349 final
    elif name == 'mst_o':
        model = MST_o.MST_Plus_Plus() # 60349 final
    else:
        print(f'Method {name} is not defined !!!!')

    return model


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        cudnn.benchmark = True  # 匹配高效算法 增加运行效率
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def mk_outfile_dir():
    # output path
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    args.outf = args.outf + date_time
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)


def main(args):

    # 初始化log
    mk_outfile_dir()
    # logging
    log_dir = os.path.join(args.outf, 'train.log')
    logger = initialize_logger(log_dir)

    # load数据
    train_dataset = TrainDataset(args.data_root)
    val_dataset = ValidDataset(args.data_root)

    epochs = args.end_epoch
    batch_size = args.batch_size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True # 提高数据从cpu到gpu的传输效率
                                               )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1,
                                             num_workers=0,
                                             pin_memory=True
                                             )


    # create model
    model = create_model(args.model_name).to(try_gpu())
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))  # 计算参数量

    # create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*epochs, eta_min=1e-6)

    # recode general info
    logger.info(f'Model:{args.model_name}, Batch_size:{batch_size}, Epoch:{epochs}, Dataset:{args.data_root}, Parameters number is {sum(param.numel() for param in model.parameters())}')

    # recode criterion
    recode = 100

    for epoch in range(0, epochs):

        losses = AverageMeter()

        criterion_mrae = Loss_MRAE()
        criterion_psnr = T_Loss_PSNR()
        criterion_ssim = T_Loss_SSIM()

        # ============================= train =============================
        model.train()
        print()
        for i, (image, target) in enumerate(train_loader):
            image = image.to(torch.float32).to(try_gpu())
            target = target.to(torch.float32).to(try_gpu())

            # 优化策略 Adam
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()  # 重置梯度
            output = model(image)  # input进模型
            loss_p = criterion_psnr(target, output)  # 计算psnr
            loss_s = criterion_ssim(target, output)  # 计算ssim

            loss = criterion_mrae(output, target) + (1 - loss_s)  # 计算loss   mrae + （1-ssim）
            loss.backward()  # 反向传播

            optimizer.step()  # 优化网络参数，如权重等
            scheduler.step()  # 优化学习率等参数
            losses.update(loss.data)

            print(f'[train epoch:{epoch + 1}/{epochs}] batch:{i + 1}/{len(train_loader)}, device:{image.device}, lr:{lr:.9f}, loss:{losses.avg:.9f}, ssim:{loss_s:.9f}, psnr:{loss_p:.9f}')


        # ============================= val =============================
        model.eval()
        print()
        losses_mrae = AverageMeter()
        losses_psnr = AverageMeter()
        losses_ssim = AverageMeter()
        for i, (image, target) in enumerate(val_loader):
            image = image.to(torch.float32).to(try_gpu())
            target = target.to(torch.float32).to(try_gpu())
            with torch.no_grad():
                output = model(image)

                # save a output img in valid
                # if i == 0 :
                #     outi(output, args.outf, epoch)

                loss_mrae = criterion_mrae(output, target)  # 计算mrae
                loss_ssim = criterion_ssim(target, output)  # 计算ssim
                loss_psnr = criterion_psnr(target, output)  # 计算psnr

            losses_mrae.update(loss_mrae.data)
            losses_ssim.update(loss_ssim)
            losses_psnr.update(loss_psnr)

        # Save model
        if (epoch % 5 == 0) or (losses_mrae.avg < recode):
            print(f'Saving to {args.outf}')
            save_checkpoint(args.outf, epoch, model, optimizer)
            if losses_mrae.avg < recode:
                recode = losses_mrae.avg

        # logging loss
        print(f'[valid epoch:{epoch + 1}/{epochs}] device:{image.device}, lr:{lr:.9f},Train Loss:{losses.avg:.9f}, Test mrae:{losses_mrae.avg:.9f}, Test ssim:{losses_ssim.avg:.9f}, Test psnr:{losses_psnr.avg:.9f}')
        logger.info(f'[valid epoch:{epoch + 1}/{epochs}] device:{image.device}, lr:{lr:.9f},Train Loss:{losses.avg:.9f}, Test mrae:{losses_mrae.avg:.9f}, Test ssim:{losses_ssim.avg:.9f}, Test psnr:{losses_psnr.avg:.9f}')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="RGB to Multispectral")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--end_epoch", type=int, default=50, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
    parser.add_argument("--outf", type=str, default='./log/', help='path log files')
    parser.add_argument("--data_root", type=str, default='./dataset/')
    parser.add_argument("--model_name", type=str, default='mst_o', help='model name')

    args = parser.parse_args()

    return args


# 'mprnet_o'
# 'hscnn_plus_o'* not enough memory
# 'hdnet_o'
# 'hrnet_o'
# 'mst_o'

if __name__ == '__main__':
    args = parse_args()
    main(args)
