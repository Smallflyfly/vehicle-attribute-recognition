#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/3/3 11:05 
"""
import os

import pandas
from torch.utils.data import Dataset

from torchvision.transforms import transforms

from utils.utils import resize_image

TYPES = ['car', 'suv', 'van', 'truck']


def get_transform(mode='train'):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.40895729, 0.43373401, 0.42956238], std=[0.20944411, 0.22798627, 0.21760352])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.40895729, 0.43373401, 0.42956238], std=[0.20944411, 0.22798627, 0.21760352])
        ])

    return transform


class KTrainDataset(Dataset):

    def __init__(self, data_root, image_path, label_path, width=448, height=448, mode='train'):
        self.data_root = data_root
        self.image_path = image_path
        self.label_path = label_path
        self.mode = mode
        self.width = width
        self.height = height
        self.label_map = {}
        self.images = []
        self.labels = []
        self.transform = get_transform(mode)

        self._read_csv()

        self._prepare_data()

    def _prepare_data(self):
        for image in self.image_path:
            label = self.label_map[image]
            self.images.append(os.path.join(self.data_root, image))
            self.labels.append(label)

    def _read_csv(self):
        dataframe = pandas.read_csv(self.label_path, 'rb', engine='python')
        all_labels = dataframe['id,type']
        for line in all_labels:
            image, label = line.strip().split(',')
            label = TYPES.index(label)
            self.label_map[image] = label

    def __getitem__(self, index):
        image = self.images[index]
        im = resize_image(image, self.width, self.height)
        im = self.transform(im)
        label = self.labels[index]
        return im, label

    def __len__(self):
        return len(self.images)
