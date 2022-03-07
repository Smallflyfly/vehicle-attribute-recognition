#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/28 15:10 
"""
import os

import pandas
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.utils import resize_image

TYPES = ['car', 'suv', 'van', 'truck']


class VehicleDataset(Dataset):

    def __init__(self, image_path, label_path, width=480, height=480):
        self.image_path = image_path
        self.label_path = label_path
        self.width = width
        self.height = height
        self.train_images = []
        self.train_labels = []
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.40895729, 0.43373401, 0.42956238], std=[0.20944411, 0.22798627, 0.21760352])
        ])

        self._read_csv()

    def _read_csv(self):
        dataframe = pandas.read_csv(self.label_path, 'rb', engine='python')
        # print(dataframe)
        all_labels = dataframe['id,type']
        for line in all_labels:
            image, label = line.strip().split(',')
            self.train_images.append(os.path.join(self.image_path, image))
            index = TYPES.index(label)
            self.train_labels.append(index)

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.train_labels[index]
        im = resize_image(image, self.width, self.height)
        im = self.transforms(im)
        return im, label

    def __len__(self):
        return len(self.train_images)