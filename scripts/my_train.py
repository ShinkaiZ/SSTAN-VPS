import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

import sys

sys.path.append('./')
from config import config
from lib.module.vacs import VACSNet as Network
from lib.dataloader.dataloader import get_video_dataset
from lib.utils.utils import clip_gradient, adjust_lr
from einops import rearrange


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, pred, mask):
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(dim=1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        smooth = 1
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        mask_flat = mask.view(size, -1)
        intersection = pred_flat * mask_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + mask_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return (wbce + dice_loss).mean()


def loss_function(outputs, target):
    b, l, _, _, _ = target.shape

    criterion_res = BCEDiceLoss().cuda()

    output_target = outputs['seg_final']
    out1 = outputs['out1']
    out2 = outputs['out2']
    out3 = outputs['out3']
    out4 = outputs['out4']
    output_target = rearrange(output_target, 'b l c h w -> (b l) c h w')
    out1 = rearrange(out1, 'b l c h w -> (b l) c h w')
    out2 = rearrange(out2, 'b l c h w -> (b l) c h w')
    out3 = rearrange(out3, 'b l c h w -> (b l) c h w')
    out4 = rearrange(out4, 'b l c h w -> (b l) c h w')
    target = rearrange(target, 'b l c h w -> (b l) c h w')
    loss0 = criterion_res(output_target, target)
    target1 = F.interpolate(target, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss1 = criterion_res(out1, target1)
    target2 = F.interpolate(target1, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss2 = criterion_res(out2, target2)
    target3 = F.interpolate(target2, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss3 = criterion_res(out3, target3)
    target4 = F.interpolate(target3, scale_factor=0.5, mode='bilinear', align_corners=True)
    loss4 = criterion_res(out4, target4)
    loss_ce = (loss0 + loss1 + loss2 + loss3 + loss4) / 5

    loss = loss_ce

    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        total_loss = F.binary_cross_entropy(pred.squeeze(), target.squeeze().float())
        return total_loss


def train(train_loader, model, optimizer, epoch, save_path, loss_func):
    global step
    model.cuda().train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()

            preds = model(images)

            loss = loss_func(preds, gts)
            loss.backward()

            clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                      format(datetime.now(), epoch, config.epoches, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.
                    format(epoch, config.epoches, i, total_step, loss.data))

        os.makedirs(os.path.join(save_path, "epoch_%d" % (epoch + 1)), exist_ok=True)
        save_root = os.path.join(save_path, "epoch_%d" % (epoch + 1))
        torch.save(model.state_dict(), os.path.join(save_root, "SSTAN.pth"))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoches, loss_all))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


if __name__ == '__main__':

    model = Network(cfg=config).cuda()

    if config.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif config.gpu_id == '0, 1':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        print('USE GPU 0 and 1')
    elif config.gpu_id == '2, 3':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        print('USE GPU 2 and 3')
    elif config.gpu_id == '0, 1, 2, 3':
        model = nn.DataParallel(model)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        print('USE GPU 0, 1, 2 and 3')

    cudnn.benchmark = True

    base_params = [params for name, params in model.named_parameters() if ("temporal_high" in name)]
    finetune_params = [params for name, params in model.named_parameters() if ("temporal_high" not in name)]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config.base_lr, 'weight_decay': 1e-4, 'name': "base_params"},
        {'params': finetune_params, 'lr': config.finetune_lr, 'weight_decay': 1e-4, 'name': 'finetune_params'}])

    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_func = loss_function

    # load data
    print('load data...')
    train_loader = get_video_dataset()
    train_loader = data.DataLoader(dataset=train_loader,
                                   batch_size=config.batchsize,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=False
                                   )
    print('Train on {}'.format(config.dataset))
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('Train on {}'.format(config.dataset))
    logging.info("Network-Train")
    print("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
                 'save_path: {}; decay_epoch: {}'.format(config.epoches, config.base_lr, config.batchsize, config.size,
                                                         config.clip,
                                                         config.decay_rate, config.save_path, config.decay_epoch))
    print('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; '
          'save_path: {}; decay_epoch: {}'.format(config.epoches, config.base_lr, config.batchsize, config.size,
                                                  config.clip,
                                                  config.decay_rate, config.save_path, config.decay_epoch))
    step = 0

    print("Start train...")
    for epoch in range(config.epoches):
        cur_lr = adjust_lr(optimizer, config.base_lr, epoch, config.decay_rate, config.decay_epoch)
        train(train_loader, model, optimizer, epoch, save_path, loss_func)
