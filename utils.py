import os
import json
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.xavier_normal_(module.weight)
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def initialize_weights2(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):  #SynchronizedBatchNorm2d
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(num_classes * label_true[mask].astype(int) +
                        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluation(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    # hist += _fast_hist(predictions.flatten(), gts.flatten(), num_classes)
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0 :gt, axis 1 : prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def prob2lines(scoremaps, out_file, num_classes=4, pts=18, thr=0.3):
    for s in range(num_classes):
        coordinate = np.zeros((1, pts))
        scoremap = scoremaps[s + 1] * 255
        # print scoremap * 255
        for i in range(pts):
            lineId = int(288 - i * 20 * 288 / 590)
            line = scoremap[lineId - 1, :]
            lid, value = np.argmax(line), np.max(line)
            if float(value) / 255 > thr:
                coordinate[0, i] = lid

        if np.sum(coordinate > 0) < 2:
            coordinate = None
        # print coordinate
        if coordinate is not None:
            for i in range(pts):
                if coordinate[0, i] > 0:
                    out_file.write(str(int(coordinate[0, i] * 1640 / 800) - 1) + ' '
                                   + str(int(590 - i * 20) - 1) + ' ')
            out_file.write('\n')
    out_file.close()


def prob2lines2(scoremaps, out_file, num_classes=4, thr=0.3):
    pts = [589, 569, 549, 529, 509, 489, 469, 449, 429, 409, 389, 369, 349, 329, 309, 289, 269, 249]
    for s in range(num_classes):
        coordinate = np.zeros((1, len(pts)))
        scoremap = scoremaps[s + 1]
        for i in range(len(pts)):
            # lineId = int(288 - i * 20 * 288 / 590)
            line = scoremap[pts[i], :]
            lid, value = np.argmax(line), np.max(line)
            if float(value) > thr:
                coordinate[0, i] = lid

        if np.sum(coordinate > 0) < 2:
            coordinate = None
        # print coordinate
        if coordinate is not None:
            for i in range(len(pts)):
                if coordinate[0, i] > 0:
                    out_file.write(str(coordinate[0, i]) + ' ' + str(pts[i]) + ' ')
            out_file.write('\n')
    out_file.close()



def ts_prob2lines(scoremaps, img_info, run_time, out_file, num_classes=4, thr=0.3):
    # scoremaps: has the same size as original image and have applied softmax function
    # num_classes: without background

    h_samples = [160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,
                410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,
                660,670,680,690,700,710]
    x = np.zeros((num_classes, len(h_samples))) - 2

    for s in range(num_classes):
        scoremap = scoremaps[s + 1]
        for i in range(len(h_samples)):
            line = scoremap[h_samples[i], :]
            line_pre = scoremap[h_samples[i]-1, :]
            line_aft = scoremap[h_samples[i]+1, :]

            lid, value = np.argmax(line), np.max(line)
            lid_pre, value_pre = np.argmax(line_pre), np.max(line_pre)
            lid_aft, value_aft = np.argmax(line_aft), np.max(line_aft)

            lid_avg = sorted([lid_pre, lid, lid_aft])[1]
            value_avg = sorted([value_pre, value, value_aft])[1]

            if float(value_avg) > thr:
                x[s, i] = lid_avg

        # if np.sum(x[s, :] > 0) < 2:
        #     x[s, :] = -2

    x = x.astype(np.int)
    x_out = []
    for i in range(x.shape[0]):
        if (x[i, :] == -2).all():
            continue
        else:
            x_out.append(x[i, :].tolist())

    info = {
        'lanes': x_out,
        'h_samples': h_samples,
        'run_time': float(run_time),
        'raw_file': img_info,
    }
    out_file.write(json.dumps(info) + '\n')

