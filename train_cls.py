# baseline, main
import torch
import os
import sys
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from model.dataset import ClassificationDataset
from model.meshmae import Mesh_baseline
from model.reconstruction import save_results
from model.utils import ClassificationMajorityVoting
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

colors = [[0, 114, 189],
          [217, 83, 26],
          [238, 177, 32],
          [126, 47, 142],
          [117, 142, 48],
          [76, 190, 238],
          [162, 19, 48],
          [240, 166, 202],
          [50, 114, 189],
          [217, 23, 26],
          [158, 177, 32],
          [126, 47, 92],
          [117, 92, 48],
          [76, 140, 238],
          [112, 19, 48],
          [190, 166, 202],
          [20, 104, 179],
          [237, 73, 6],
          [258, 167, 12],
          [146, 37, 122],
          [137, 132, 28],
          [96, 180, 218],
          [182, 9, 28],
          [250, 156, 192],
          [70, 104, 169],
          [137, 13, 6],
          [178, 167, 12],
          [146, 17, 72],
          [137, 82, 28],
          [96, 130, 218],
          [132, 9, 28],
          [210, 156, 182],
          [222, 156, 192],
          [40, 104, 179],
          [197, 23, 23],
          [158, 197, 2],
          [106, 47, 52],
          [117, 52, 98],
          [76, 110, 238],
          [112, 99, 8],
          [110, 196, 12]]


def seed_torch(seed=12):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_embedding(data, label, title):
    """
    :param data:数据集
    :param label:样本标签
    :param title:图像标题
    :return:图像
    """
    print(len(label))
    cmap = cm.rainbow(np.linspace(0, 1, len(label)))

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    # ax = plt.subplot(111)
    for i in range(data.shape[0]):
        c = cm.rainbow(int(255 / 40 * label[i]))
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 40),  fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], color=c, alpha=0.5)
    plt.xticks()  # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=9)
    # 返回值
    return fig


def train(net, optim, scheduler, names, criterion, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0

    for it, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, labels, mesh_paths) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()
        labels = labels.cuda()
        n_samples += faces.shape[0]
        outputs = net(faces, feats, centers, Fs, cordinates)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

    scheduler.step()
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples
    print('epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}'.format(names, epoch, epoch_loss, epoch_acc))
    message = 'epoch ({:}): {:} Train Loss: {:.4f} Acc: {:.4f}'.format(names, epoch, epoch_loss, epoch_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)

def test(net, names, criterion, test_dataset, epoch, args):
    ts = TSNE(n_components=2, init='pca', random_state=0)

    # for net_ in net:
    #     net_.eval()
    #     voted.append(ClassificationMajorityVoting(args.n_classes))
    net.eval()
    voted = ClassificationMajorityVoting(args.n_classes)

    running_loss = 0
    running_corrects = 0
    n_samples = 0

    for i, (feats_patch, center_patch, coordinate_patch, face_patch, np_Fs, labels, mesh_paths) in enumerate(
            test_dataset):
        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()

        labels = labels.cuda()

        n_samples += faces.shape[0]
        batch_size = faces.shape[0]
        with torch.no_grad():
            outputs = net(faces, feats, centers, Fs, cordinates).detach()
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * faces.size(0)
            voted.vote(mesh_paths, preds, labels)

    epoch_acc = running_corrects.double() / n_samples
    epoch_loss = running_loss / n_samples
    epoch_vacc = voted.compute_accuracy()
    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        #torch.save(best_model_wts, os.path.join('checkpoints', names, f'acc-{epoch_acc:.4f}-{epoch:.4f}.pkl'))
        torch.save(best_model_wts, os.path.join('checkpoints', names, 'best_acc.pkl'))
    if test.best_vacc < epoch_vacc:
        test.best_vacc = epoch_vacc
        best_model_wts = copy.deepcopy(net.state_dict())
        #torch.save(best_model_wts, os.path.join('checkpoints', names, f'vacc-{epoch_vacc:.4f}-{epoch:.4f}.pkl'))
        torch.save(best_model_wts, os.path.join('checkpoints', names, 'best_vacc.pkl'))

    message = 'epoch ({:}): {:} test Loss: {:.4f} Acc: {:.4f} Best Acc: {:.4f} '.format(names, epoch, epoch_loss,
                                                                                       epoch_acc,
                                                                                       test.best_acc)
    message += 'test acc [voted] = {:} Best acc [voted] = {:}'.format(epoch_vacc, test.best_vacc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)
    print(message)


if __name__ == '__main__':
    seed_torch(seed=43)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=str, default=None)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, required=True, default=500)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    parser.add_argument('--channels', type=int, default=10)
    args = parser.parse_args()
    mode = args.mode
    args.name = args.name
    dataroot = args.dataroot

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')
    train_dataset = ClassificationDataset(dataroot, train=True, augment=augments)
    test_dataset = ClassificationDataset(dataroot, train=False)
    print(len(train_dataset))
    print(len(test_dataset))

    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ========== Network ==========
    net = Mesh_baseline(decoder_embed_dim=args.decoder_dim,
                        masking_ratio=args.mask_ratio,
                        encoder_depth=args.encoder_depth,
                        num_heads=args.heads,
                        channels=args.channels,
                        patch_size=args.patch_size,
                        embed_dim=args.dim,
                        decoder_num_heads=args.decoder_num_heads,
                        decoder_depth=args.decoder_depth).to(device)

    # ========== Optimizer ==========
    if args.optim.lower() == 'adam':
        optim = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim.lower() == 'sgd':
        optim = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones.lower() != 'none':
        ms = args.lr_milestones
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optim, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(args.num_warmup_steps),
                                                    num_training_steps=args.n_epoch)

    print(scheduler)
    criterion = nn.CrossEntropyLoss()
    checkpoint_names = []
    checkpoint_path = os.path.join('checkpoints', args.name)

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint.lower() != 'none':
        net.load_state_dict(torch.load(args.checkpoint), strict=False)

    train.step = 0
    test.best_acc = 0
    test.best_vacc = 0

    # ========== Start Training ==========

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch(epoch)
            print('epoch', epoch)
            train(net, optim, scheduler, args.name, criterion, train_data_loader, epoch, args)
            print('train finished')
            test(net, args.name, criterion, test_data_loader, epoch, args)
            print('test finished')



    else:

        test(net, args.name, criterion, test_data_loader, 0, args)
