import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255, aux_weight=None, print_aux=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.aux_weight = aux_weight
        self.print_aux = print_aux
        self.criterion1 = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index)

    def forward(self, predicts, target):
        h, w = target.size(1), target.size(2)
        # print len(predicts)

        aux1 = F.upsample(predicts[1], size=(h, w), mode='bilinear', align_corners=True)
        aux2 = F.upsample(predicts[2], size=(h, w), mode='bilinear', align_corners=True)
        aux3 = F.upsample(predicts[3], size=(h, w), mode='bilinear', align_corners=True)
        aux4 = F.upsample(predicts[4], size=(h, w), mode='bilinear', align_corners=True)

        loss = self.criterion1(predicts[0], target)
        loss1 = self.criterion1(aux1, target)
        loss2 = self.criterion1(aux2, target)
        loss3 = self.criterion1(aux3, target)
        loss4 = self.criterion1(aux4, target)

        total_loss = loss + self.aux_weight * (loss1 + loss2 + loss3 + loss4)
        if self.print_aux:
            return total_loss, loss
        return total_loss

