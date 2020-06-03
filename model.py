import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dcp.modules import ConvOffset2d

from resnet import resnet101
from sync_batchnorm import SynchronizedBatchNorm2d
from config import res101_csail_path, res101_path

norm_layer = SynchronizedBatchNorm2d
relu = nn.ReLU(inplace=True)


class SGCA(nn.Module):
    def __init__(self, down_dim, lateral_dim):
        super(SGCA, self).__init__()
        self.down_dim = down_dim
        self.lateral_dim = lateral_dim
        self.threshold = int(self.down_dim * 0.5)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.down_conv = nn.Conv2d(down_dim, lateral_dim, kernel_size=1, bias=False)
        self.lateral_conv = nn.Conv2d(lateral_dim, lateral_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x_down, x_lateral):
        b, c, h, w = x_down.size()
        att_down = self.global_pool(x_down).view(b, c)
        abs_down = torch.abs(att_down)
        r = random.random()
        key = r ** (1 / abs_down)
        middle = torch.sort(key, dim=1)[0][:, self.threshold: self.threshold + 1]
        mask = key >= middle
        mask = mask.float().contiguous().view(b, c, 1, 1)

        x_down_chosen = x_down * mask
        out_down = self.global_pool(x_down_chosen).view(b, c, 1, 1)
        out_down = self.down_conv(out_down)

        out_lateral = self.lateral_conv(x_lateral)
        out = out_lateral * out_down

        return out + x_lateral


class DFConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(DFConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.offset = nn.Conv2d(self.in_dim, 2 * self.kernel_size * self.kernel_size,
                                kernel_size=self.kernel_size,
                                padding=(self.kernel_size - 1) // 2)
        self.conv_offset = ConvOffset2d(self.in_dim, self.out_dim,
                                        kernel_size=self.kernel_size,
                                        padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        offset = self.offset(x)
        conv_offset = self.conv_offset(x, offset)
        return conv_offset


class PDC(nn.Module):
    def __init__(self, in_dim, out_dim, setting):
        super(PDC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.setting = setting

        self.features = []
        for s in self.setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(out_dim),
                relu,
                DFConv(out_dim, out_dim, kernel_size=3),
                norm_layer(out_dim),
                relu,
            ))
        self.features = nn.ModuleList(self.features)

        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            relu,
        )

    def forward(self, x):
        out = self.residual(x)
        for f in self.features:
            out = out + F.upsample(f(x), x.size()[2:], mode='bilinear')
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.norm1 = norm_layer(in_channels)

        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = norm_layer(in_channels)

        self.conv3 = nn.Conv2d(in_channels, n_filters, 1)
        self.norm3 = norm_layer(n_filters)
        self.relu = relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


class Baseline(nn.Module):
    def __init__(self, num_classes, deep_base=True):
        super(Baseline, self).__init__()
        resnet = resnet101(pretrained=False, deep_base=deep_base, norm_layer=norm_layer)

        if deep_base:
            resnet.load_state_dict(torch.load(res101_csail_path))
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        else:
            resnet.load_state_dict(torch.load(res101_path))
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.sgca_3 = SGCA(2048, 1024)
        self.sgca_2 = SGCA(1024, 512)
        self.sgca_1 = SGCA(512, 256)

        self.pdc_4 = PDC(2048, 128, (5, 7, 9))
        self.pdc_3 = PDC(1024, 128, (10, 14, 18))
        self.pdc_2 = PDC(512, 128, (20, 28, 36))
        self.pdc_1 = PDC(256, 128, (40, 56, 72))

        self.decoder4 = DecoderBlock(128, 128)
        self.decoder3 = DecoderBlock(128, 128)
        self.decoder2 = DecoderBlock(128, 128)
        self.decoder1 = DecoderBlock(128, 128)

        self.pred = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            relu,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            relu,
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )

        self.aux1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.aux4 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        m1_3 = self.sgca_3(layer4, layer3)
        m1_2 = self.sgca_2(m1_3, layer2)
        m1_1 = self.sgca_1(m1_2, layer1)

        m2_4 = self.pdc_4(layer4)
        m2_3 = self.pdc_3(m1_3)
        m2_2 = self.pdc_2(m1_2)
        m2_1 = self.pdc_1(m1_1)

        if self.training:
            aux1 = self.aux1(m2_1)
            aux2 = self.aux2(m2_2)
            aux3 = self.aux3(m2_3)
            aux4 = self.aux4(m2_4)

        decoder4 = self.decoder4(m2_4) + m2_3
        decoder3 = self.decoder3(decoder4) + m2_2
        decoder2 = self.decoder2(decoder3) + m2_1
        decoder1 = self.decoder1(decoder2)

        pred = self.pred(decoder1)
        out = [pred]
        if self.training:
            out.append(aux1)
            out.append(aux2)
            out.append(aux3)
            out.append(aux4)
            return out
        return out

