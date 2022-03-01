#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/3/1 9:54 
"""
import logging
import os
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet101

from dataset.dataset import VehicleDataset
from utils.utils import build_optimizer, build_scheduler
import tensorboardX as tb
import numpy as np


EPOCH = 200
BATCH_SIZE = 1
K_FOLD = 10

t = time.strftime("%Y%m%d%H%M%S", time.localtime())
train_log = 'logs/' + t + '.log'
logging.basicConfig(filename=train_log, format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ',
                    level=logging.INFO)


def run_train():
    dataset = VehicleDataset('data/train', 'data/train_sorted.csv', 448, 448)
    train_dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    # model = inception(num_classes=4)
    model = resnet101(num_classes=4)
    model = model.cuda()
    loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, lr_scheduler='cosine', max_epoch=EPOCH)
    writer = tb.SummaryWriter()
    logger = logging.getLogger()
    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    logger.addHandler(KZT)


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def train():



if __name__ == '__main__':
    train()