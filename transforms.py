import math
import random
import torch
import numpy as np
from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size

        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class DeNormalize(object):
    def __int__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class RandomScale(object):
    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        tsize = int(np.random.choice([0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.0]) * min(w, h))
        if (w <= h and w == tsize) or (h <= w and h == tsize):
            return img, mask
        if w < h:
            ow = tsize
            oh = int(tsize * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = tsize
            ow = int(tsize * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedRatio(object):
    def __init__(self, minW, maxW, minH, maxH):
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH

    def __call__(self, img, mask):
        assert img.size == mask.size

        targetW = random.randint(self.minW, self.maxW)
        targetH = random.randint(self.minH, self.maxH)
        return img.resize((targetW, targetH), Image.BICUBIC), mask.resize((targetW, targetH), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, minW, maxW, minH, maxH):
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH

    def __call__(self, img, mask):
        targetW = random.randint(self.minW, self.maxW)
        targetH = random.randint(self.minH, self.maxH)

        inputW, inputH = img.size

        if inputH == targetH and inputW == targetW:
            return img, mask
        if inputH < targetH or inputW < targetW:
            delta_w = targetW - inputW
            delta_h = targetH - inputH
            padding = (int(delta_w / 2), int(delta_h / 2), delta_w - int(delta_w / 2), delta_h - int(delta_h / 2))
            img = ImageOps.expand(img, border=padding, fill=0)
            mask = ImageOps.expand(mask, border=padding, fill=0)
            return img, mask

        x1 = random.randint(0, inputW - targetW)
        y1 = random.randint(0, inputH - targetH)
        return img.crop((x1, y1, x1+targetW, y1+targetH)), mask.crop((x1, y1, x1+targetW, y1+targetH))


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size

        inputW, inputH = img.size
        [w, h] = self.size
        if inputW == w and inputH == h:
            return img, mask
        if inputW < w or inputH < h:
            delta_w = w - inputW
            delta_h = h - inputH
            padding = (int(delta_w / 2), int(delta_h / 2), delta_w - int(delta_w / 2), delta_h - int(delta_h / 2))
            img = ImageOps.expand(img, border=padding, fill=0)
            mask = ImageOps.expand(mask, border=padding, fill=0)
            return img, mask

        x1 = random.randint(0, inputW - w)
        y1 = random.randint(0, inputH - h)
        return img.crop((x1, y1, x1 + w, y1 + h)), mask.crop((x1, y1, x1 + w, y1 + h))


