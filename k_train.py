#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/3/1 9:54 
"""
import logging
import os
import time

import numpy as np
import tensorboardX as tb
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet101

from dataset.k_train_dataset import KTrainDataset
from utils.utils import build_optimizer, build_scheduler, load_pretrained_weight, seed_it

EPOCH = 200
BATCH_SIZE = 8
K_FOLD = 10

t = time.strftime("%Y%m%d%H%M%S", time.localtime())
train_log = 'logs/' + t + '.log'
logging.basicConfig(filename=train_log, format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)

IMAGE_PATH = 'data/train'
writer = tb.SummaryWriter()


def val_model(model, val_dataloader):
    model.eval()
    softmax = nn.Softmax()
    sum = 0
    for index, data in enumerate(val_dataloader):
        image, label = data
        image = image.cuda()
        out = model(image)
        out = softmax(out).cpu().detach().numpy()
        id = np.argmax(out)
        sum += 1 if id == label.numpy()[0] else 0
    return sum / len(val_dataloader)


def run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold):
    best_acc = 0
    for epoch in range(1, EPOCH+1):
        model.train()
        for i, data in enumerate(train_dataloader):
            image, label = data
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            out = model(image)
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                logging.info(' Fold:{} Epoch:{}({}/{}) lr:{:6f} loss:{:6f}:'.format(
                    fold + 1, epoch, i, EPOCH, optimizer.param_groups[-1]['lr'], loss.item()))

            index = fold * EPOCH * len(train_dataloader) + epoch * len(train_dataloader) + i + 1
            if index % 20 == 0:
                writer.add_scalar('loss', loss, index)
                writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], index)

        scheduler.step()
        val_acc = val_model(model, val_dataloader)
        writer.add_scalar('val_acc', val_acc, fold * EPOCH + epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'weights/best_k_train_{}_fold.pth'.format(fold+1))
            logging.info('Best epoch/fold: {}/{} best acc: {:6f}'.format(epoch, fold+1, best_acc))
        logging.info('Best acc: {:6f}'.format(best_acc))


def train():
    seed = 2022
    seed_it(seed)
    images = os.listdir(IMAGE_PATH)
    images = np.array(images)
    folds = KFold(n_splits=K_FOLD, shuffle=True, random_state=seed).split(range(len(images)))
    model = resnet101(num_classes=4)
    load_pretrained_weight(model, 'weights/resnet101-5d3b4d8f.pth')
    model = model.cuda()
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCH)
    # log
    logger = logging.getLogger()
    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    logger.addHandler(KZT)

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_dataset = KTrainDataset('data/train', images[train_idx], 'data/train_sorted.csv', width=256, height=256, mode='train')
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
        val_dataset = KTrainDataset('data/train', images[val_idx], 'data/train_sorted.csv', width=256, height=256, mode='val')
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=0, shuffle=False)
        run_train(model, train_dataloader, val_dataloader, loss_func, optimizer, scheduler, fold)

    writer.close()


if __name__ == '__main__':
    train()