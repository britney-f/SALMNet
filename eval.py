import os
import numpy as np
import datetime
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as transforms

from datasets import culane
import transforms as extend_transforms
from utils import check_mkdir, AverageMeter, evaluation, prob2lines
from model import Baseline
from torch.backends import cudnn

cudnn.enable = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ckpt_path = './ckpt'
exp_name = 'model'

args = {
    'checkpoint':'60000',
    'val_size': [800, 288],
    'save_results': True,
    'deep_base': True,
}

mean = [0.3598, 0.3653, 0.3662]
std = [0.2573, 0.2663, 0.2756]

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
mask_transform = extend_transforms.MaskToTensor()
to_pil = transforms.ToPILImage()
val_joint_transform = extend_transforms.Scale(args['val_size'])


criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.4, 1, 1, 1, 1]).cuda(), size_average=True,
                                      ignore_index=culane.ignore_label)
criterion = criterion.cuda()


def main():
    net = Baseline(num_classes=culane.num_classes, deep_base=args['deep_base']).cuda()

    print('load checkpoint \'%s.pth\' for evaluation' % args['checkpoint'])
    pretrained_dict = torch.load(os.path.join(ckpt_path, exp_name, args['checkpoint'] + '_checkpoint.pth'))
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict)

    net.eval()

    save_dir = os.path.join(ckpt_path, exp_name, 'vis_%s_test' % args['checkpoint'])
    check_mkdir(save_dir)
    log_path = os.path.join(save_dir, str(datetime.datetime.now()) + '.log')

    data_list = [l.strip('\n') for l in open(os.path.join(culane.root, culane.list, 'test_gt.txt'), 'r')]

    loss_record = AverageMeter()
    gt_all, prediction_all=[], []

    for idx in range(len(data_list)):
        print('evaluating %d / %d' % (idx + 1, len(data_list)))

        img = Image.open(culane.root + data_list[idx].split(' ')[0]).convert('RGB')
        gt = Image.open(culane.root + data_list[idx].split(' ')[1])

        img, gt = val_joint_transform(img, gt)

        with torch.no_grad():
            img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
            gt_var = Variable(mask_transform(gt).unsqueeze(0)).cuda()

            prediction = net(img_var)[0]

            loss = criterion(prediction, gt_var)
            loss_record.update(loss.data, 1)

            scoremap = F.softmax(prediction, dim=1).data.squeeze().cpu().numpy()

            prediction = prediction.data.max(1)[1].squeeze().cpu().numpy().astype(np.uint8)
            prediction_all.append(prediction)
            gt_all.append(np.array(gt))

        if args['save_results']:
            check_mkdir(save_dir + data_list[idx].split(' ')[0][:-10])
            out_file = open(os.path.join(save_dir, data_list[idx].split(' ')[0][1:-4] + '.lines.txt'), 'w')
            prob2lines(scoremap, out_file)

    acc, acc_cls, mean_iu, fwavacc = evaluation(prediction_all, gt_all, culane.num_classes)
    log = 'val results: loss %.5f  acc %.5f  acc_cls %.5f  mean_iu %.5f  fwavacc %.5f' % \
              (loss_record.avg, acc, acc_cls, mean_iu, fwavacc)
    print(log)
    open(log_path, 'w').write(log + '\n')


if __name__ == '__main__':
    main()

