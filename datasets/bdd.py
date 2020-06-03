import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
# import matplotlib as plt

root = '/home/hdd4t/YuTianfei/BDD100K'
list = 'list'

num_classes = 2
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
    assert mode in ['train', 'val', 'test']

    img_list = [l.strip('\n') for l in open(os.path.join(
        root, list, mode + '_gt_bdd.txt'), 'r')]
    print('The size of dataset is %d.' % len(img_list))

    files = []
    for name in img_list:
        # print(img_name, mask_name)
        img_path = root + name.split(' ')[0]
        mask_path = root + name.split(' ')[1]
        files.append({
            'img': img_path,
            'mask': mask_path,
            'info': name
        })

    return files


class BDD(data.Dataset):
    def __init__(self, mode, joint_transform = None, transform = None, mask_transform =None):
        self.imgs = make_dataset(mode)
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.mask_transform = mask_transform
        # self.id_to_trainid = {9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8}

    def __getitem__(self, index):
        if self.mode == 'test':
            img = Image.open(self.imgs[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img

        img_path = self.imgs[index]['img']
        img = Image.open(img_path).convert('RGB')

        mask_path = self.imgs[index]['mask']
        mask = Image.open(mask_path).convert('L')

        mask = np.array(mask)
        mask = mask / 255
        # print(np.max(mask.astype(np.int32)))
        mask = Image.fromarray(mask.astype(np.int32))

        # mask = np.array(mask)
        # mask_copy = mask.copy()
        # for k, v in self.id_to_trainid.items():
        #     mask_copy[mask == k] = v
        # mask = Image.fromarray(mask_copy.astype(np.uint8))
        # print (np.max(np.array(mask)))

        info = self.imgs[index]['info']

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return img, mask, info


    def __len__(self):
        return len(self.imgs)


