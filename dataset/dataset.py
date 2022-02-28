#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/28 15:10 
"""
import os

import cv2
import pandas
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

TYPES = ['car', 'suv', 'van', 'truck']


def _resize_image(im, width, height):
    w, h = im.shape[1], im.shape[0]
    r = min(width / w, height / h)
    new_w, new_h = int(w * r), int(h * r)
    im = cv2.resize(im, (new_w, new_h))
    pw = (width - new_w) // 2
    ph = (height - new_h) // 2
    top, bottom = ph, ph
    left, right = pw, pw

    if top + bottom + new_h < height:
        bottom += 1

    if left + right + new_w < width:
        right += 1

    im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=114)
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return im


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
        ids, types = dataframe['id'], dataframe['type']
        for id, type in zip(ids, types):
            self.train_images.append(os.path.join(self.image_path, id))
            index = TYPES.index(type)
            self.train_labels.append(index)

    def __getitem__(self, index):
        image = self.train_images[index]
        label = self.train_labels[index]
        im = _resize_image(image, self.width, self.height)
        return im, label