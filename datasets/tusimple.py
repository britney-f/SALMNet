import os
import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils import data

root = '/usr/TianfeiYu/Datasets/Tusimple'
list = 'list'


num_classes = 6
ignore_label = 255

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset(mode):
    assert mode in ['train', 'trainval', 'test']

    data_list = [l.strip('\n') for l in open(os.path.join(root, list, mode + '.txt'), 'r')]
    files = []
    for data_name in data_list:
        img_path = os.path.join(root, 'train_set' + data_name.split(' ')[0])
        mask_path = root + data_name.split(' ')[1]
        files.append({
            'img': img_path,
            'mask': mask_path,
            'info': data_name.split(' ')[0]
        })
    return files


class TUSIMPLE(data.Dataset):
    def __init__(self, mode, joint_transform=None, transform=None, mask_transform=None):
        self.imgs = make_dataset(mode)
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]['img']
        mask_path = self.imgs[index]['mask']
        info = self.imgs[index]['info']

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask, info

    def __len__(self):
        return len(self.imgs)