import os
import datetime
import time

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from datasets import culane
from model import Baseline
import transforms as extend_transforms
from loss import CrossEntropyLoss2d
from utils import check_mkdir, AverageMeter
from sync_batchnorm import DataParallelWithCallback

torch.manual_seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

args = {
    'max_iter':60000,
    'start_iter':0,
    'train_batch_size':10,
    'base_lr':0.1,
    'power':0.9,
    # 'weight_decay':5e-4,
    'momentum':0.9,
    'train_crop_size':[800,288],
    'checkpoint':'',
    'save_interval': 10000,
    'rotate_degree':10,
    'deep_base': True,
    'aux_weight': 0.1,
    'print_aux':True,
}

ckpt_path = './ckpt'
exp_name = 'model'

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.3598, 0.3653, 0.3662], [0.2573, 0.2663, 0.2756])
])
mask_transform = extend_transforms.MaskToTensor()

train_joint_transform = extend_transforms.Compose([
    extend_transforms.RandomScale(),
    extend_transforms.RandomSizedRatio(760, 842, 274, 304),
    extend_transforms.RandomRotate(args['rotate_degree']),
    extend_transforms.RandomCrop(args['train_crop_size']),
])

train_set = culane.CULANE('train', joint_transform=train_joint_transform,
                          transform=img_transform, mask_transform=mask_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=10, shuffle=True)

criterion = CrossEntropyLoss2d(weight=torch.Tensor([0.4, 1, 1, 1, 1]).cuda(), size_average=True,
                               ignore_index=culane.ignore_label, aux_weight=args['aux_weight'],
                               print_aux=args['print_aux'])
criterion = criterion.cuda()

writer = SummaryWriter(os.path.join(ckpt_path, exp_name, 'tboard'))
log_path = os.path.join(ckpt_path, exp_name, 'train' + str(datetime.datetime.now()) + '.txt')


def main():
    net = Baseline(num_classes=culane.num_classes, deep_base=args['deep_base']).cuda().train()
    net = DataParallelWithCallback(net)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['base_lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['base_lr']}],
        momentum=args['momentum'])

    if len(args['checkpoint']) > 0:
        print('training resumes from \'%s\'' % args['checkpoint'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['checkpoint'] + '_checkpoint.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['checkpoint'] + '_checkpoint_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['base_lr']
        optimizer.param_groups[1]['lr'] = args['base_lr']

    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')

    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['start_iter']

    while True:
        total_loss_record, loss_record = AverageMeter(), AverageMeter()
        for i, data in enumerate(train_loader):
            start = time.time()

            optimizer.param_groups[0]['lr'] = 2 * args['base_lr'] * (1 - float(curr_iter) / args['max_iter']
                                                                     ) ** args['power']
            optimizer.param_groups[1]['lr'] = args['base_lr'] * (1 - float(curr_iter) / args['max_iter']
                                                                     ) ** args['power']

            inputs, labels, _ = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            total_loss, loss = criterion(outputs, labels)

            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data, batch_size)
            loss_record.update(loss.data, batch_size)

            curr_iter += 1
            end = time.time()

            log = 'iter %d  lr %.13f  total_loss %.5f(%.5f)  loss %.5f(%.5f)  time %.5f' % \
                  (curr_iter, optimizer.param_groups[1]['lr'], total_loss.data, total_loss_record.avg, loss.data,
                   loss_record.avg, end-start)
            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter % 100 == 0:
                writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), curr_iter)

            if curr_iter % args['save_interval'] == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d_checkpoint.pth') % curr_iter)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, '%d_checkpoint_optim.pth' % curr_iter))

            if curr_iter > args['max_iter']:
                return


if __name__ == '__main__':
    main()


