from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
import random
import numpy as np
from models.retinaface import RetinaFace
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
# parser.add_argument('--resume_net', default='./weights/mobilenet0.25_Final.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')   # 用于学习率调整
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(2)  # 1

# 创建保存模型的文件夹 './weights/'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# ============================== 读取config ==============================
# 读取 config，如batch size, epoch , ngpu等超参数
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50
rgb_mean = (104, 117, 123)  # bgr order， 该值通过imagenet数据集统计得到
num_classes = 2
img_dim = cfg['image_size']  # 640
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']  # bool， True or False

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset   # label.txt
save_folder = args.save_folder   # 保存模型weights的文件夹

# ============================== 创建模型 ==============================
net = RetinaFace(cfg=cfg)

# 是否断点续训练
if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

# 把模型放到gpu上
device = "cuda" if gpu_train and torch.cuda.is_available() else "cpu"
if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
elif gpu_train:
    net.to(device)

# PriorBox与anchor非常类似，生成一系列候选框
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)
# ============================== 创建优化器，损失函数 ==============================
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
# 0.35是正负bbox的阈值；
# 7是负：正样本比例，正样本数量通过0.35确定，负样本通过正样本*7确定


def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    # 构建数据集 训练集有  12880张图片 及其标签
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    # 计算总迭代次数
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    # 计算起始Iteration
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # 用于学习率调整， 当cfg['decay1']时， step_index会加1， 即类似multistep中的stone一样，到该点会执行某些操作
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)  # 87400, 101200; 190epoch, 220epoch
    step_index = 0

    # 主循环
    for iteration in range(start_iter, max_iter):
        # 初始化数据迭代器，dataloader； 并保存模型
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1
        ################
        # if iteration % epoch_size < epoch_size*0.98:
        #     continue
        ################
        load_t0 = time.time()

        # 调整学习率， # 190乘以0.1， 220乘以0.001， 其余时候step_index=0，lr*1，因此不会改变
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)  # shape is (bs, 3, 640, 640)
        targets = [anno.to(device) for anno in targets]  # (n, 15) n表示目标数，15 = 4 + 1 + 10

        # forward
        out = net(images)
        # torch.Size([28, 16800, 4])   bs, anchors_all, bbox
        # torch.Size([28, 16800, 2])   bs, anchors_all, cls
        # torch.Size([28, 16800, 10])  bs, anchors_all, landmark

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()

        # 计算耗时， 打印loss信息
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))  # 当190,220epoch的时候下降 step_index一直是0，当190epoch时才会+1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
