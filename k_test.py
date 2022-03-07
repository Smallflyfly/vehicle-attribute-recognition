#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/3/4 10:58 
"""
import csv
import os

from torch import nn
from torchvision.models import resnet101
from torchvision.transforms import transforms
import numpy as np

from utils.utils import resize_image, load_pretrained_weight

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40895729, 0.43373401, 0.42956238], std=[0.20944411, 0.22798627, 0.21760352])
])

TEST_PATH = 'data/testA'
K_FOLD = 3
TYPES = ['car', 'suv', 'van', 'truck']
CSV_FILE = 'result.csv'


def create_csv(all_results):
    with open(CSV_FILE, 'wt', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        rows = ['id', 'type']
        csv_writer.writerow(rows)
        csv_writer.writerows(all_results)


def test():
    images = os.listdir(TEST_PATH)
    model = resnet101(num_classes=4)
    model = model.cuda()
    softmax = nn.Softmax()
    all_results = []
    model.eval()
    for idx, image in enumerate(images):
        im = resize_image(os.path.join(TEST_PATH, image), 256, 256)
        im = transform(im)
        im = im.cuda().unsqueeze(0)
        max_prob = 0
        index = 0
        for fold in range(1, K_FOLD+1):
            weight = 'weights/best_k_train_{}_fold.pth'.format(fold)
            load_pretrained_weight(model, weight)
            out = model(im)
            out = softmax(out).cpu().detach().numpy()[0]
            i = np.argmax(out, axis=0)
            prob = out[i]
            if max_prob < prob:
                max_prob = prob
                index = i
        all_results.append([image, TYPES[index]])
        print('{} / {}'.format(idx, len(images)))

    create_csv(all_results)


if __name__ == '__main__':
    test()