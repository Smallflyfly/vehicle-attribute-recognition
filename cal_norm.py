#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/2/22 20:01 
"""
import os
import numpy as np
import cv2

ROOT_PATH = 'data/train'


def cal():
    mean, std = None, None
    images = os.listdir(ROOT_PATH)
    for image in images:
        im = cv2.imread(os.path.join(ROOT_PATH, image))
        im = im[:, :, ::-1] / 255.
        if mean is None and std is None:
            mean, std = cv2.meanStdDev(im)
        else:
            mean_, std_ = cv2.meanStdDev(im)
            mean_stack = np.stack((mean, mean_), axis=0)
            std_stack = np.stack((std, std_), axis=0)
            mean = np.mean(mean_stack, axis=0)
            std = np.mean(std_stack, axis=0)
    return mean.reshape((1, 3))[0], std.reshape((1, 3))[0]


if __name__ == '__main__':
    res = cal()
    print(res)
    # [0.40895729, 0.43373401, 0.42956238], [0.20944411, 0.22798627, 0.21760352]