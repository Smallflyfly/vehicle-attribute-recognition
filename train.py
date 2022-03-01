#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/23 16:10 
"""
import logging
import time

import tensorboardX as tb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet101

from dataset.dataset import VehicleDataset
# from model.inception_iccv import inception
from utils.utils import build_optimizer, build_scheduler, load_pretrained_weight

EPOCH = 200
BATCH_SIZE = 1

t = time.strftime("%Y%m%d%H%M%S", time.localtime())
train_log = 'logs/' + t + '.log'
logging.basicConfig(filename=train_log, format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def train():
    dataset = VehicleDataset('data/train', 'data/train_sorted.csv', 448, 448)
    train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    # model = inception(num_classes=4)
    model = resnet101(num_classes=4)
    load_pretrained_weight(model, 'weights/resnet101-5d3b4d8f.pth')
    model = model.cuda()
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCH)
    writer = tb.SummaryWriter()
    logger = logging.getLogger()
    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    logger.addHandler(KZT)
    for epoch in range(1, EPOCH+1):
        model.train()
        for index, data in enumerate(train_dataloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            # out1, out2, out3, out4 = model(image)
            # loss1 = loss_func(out1, label)
            # loss2 = loss_func(out2, label)
            # loss3 = loss_func(out3, label)
            # loss4 = loss_func(out4, label)
            # loss = loss1 + loss2 + loss3 + loss4
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            if index % 50 == 0:
                tt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                info = 'Time: {}, Epoch: [{}/{}] [{}/{}]\n'.format(tt, epoch, EPOCH, index + 1,
                                                                   len(train_dataloader), loss)
                # info_loss1 = '=============>loss1: {:.6f}\n'.format(loss1)
                # info_loss2 = '=============>loss2: {:.6f}\n'.format(loss2)
                # info_loss3 = '=============>loss3: {:.6f}\n'.format(loss3)
                # info_loss4 = '=============>loss4: {:.6f}\n'.format(loss4)
                info_loss = '=============>total_loss: {:.6f}\n'.format(loss)
                info = info + info_loss
                logger.info(info)

            count = epoch * len(train_dataloader) + index
            if count % 20 == 0:
                # writer.add_scalar('loss1', loss1, count)
                # writer.add_scalar('loss2', loss2, count)
                # writer.add_scalar('loss3', loss3, count)
                # writer.add_scalar('loss4', loss4, count)
                writer.add_scalar('loss', loss, count)

        scheduler.step()

        if epoch % 20 == 0:
            torch.save(model.state_dict(), 'weights/resnet_{}.pth'.format(epoch))
        if epoch == EPOCH:
            torch.save(model.state_dict(), 'weights/resnet_last.pth')

    writer.close()


if __name__ == '__main__':
    train()