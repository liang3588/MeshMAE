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
from model.dataset import ShapeNetDataset
from model.meshmae import Mesh_mae
from model.reconstruction import save_results
from model.utils import ClassificationMajorityVoting
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
import time


def train(net, optim, scheduler, names, train_dataset, epoch, args):
    net.train()
    running_loss = 0

    n_samples = 0

    for it, (feats_patch, center_patch,coordinate_patch, face_patch,  np_Fs, label, mesh_paths) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.to(torch.float32).cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        n_samples += faces.shape[0]
        loss = net(faces, feats, centers, Fs, cordinates)
        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)

    epoch_loss = running_loss / n_samples


    scheduler.step()
    if train.best_loss > epoch_loss:
        train.best_loss = epoch_loss
        train.best_epoch = epoch
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('checkpoints', names, f'loss-{epoch_loss:.4f}-{epoch:.4f}.pkl'))
    print('epoch ({:}): {:} Train Loss: {:.4f}'.format(names, epoch, epoch_loss))


def test(net,  names, test_dataset, epoch, args):
    #######################################################################
    # if you are going to show the reconstruct shape, please using the following codes
    #######################################################################
    net.eval()
    for it, (feats_patch, center_patch, face_patch, coordinate_patch, np_Fs, label, mesh_paths) in enumerate(
            test_dataset):

        faces = face_patch.cuda()
        feats = feats_patch.cuda()
        centers = center_patch.cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.cuda()

        with torch.no_grad():
            loss, masked_indices, unmasked_indices, pred_vertices_coordinates, cordinates = net(faces, feats, centers, Fs, cordinates)
        save_results(masked_indices, unmasked_indices, pred_vertices_coordinates, cordinates, mesh_paths)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lr_milestones', type=str, default=None)
    parser.add_argument('--num_warmup_steps', type=str, default=None)

    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--n_dropout', type=int, default=1)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--weight', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--no_center_diff', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--mask_ratio', type=float, default=0.25)

    args = parser.parse_args()
    mode = args.mode
    dataroot = args.dataroot

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')

    if 'ShapeNet' in dataroot:
        train_dataset = ShapeNetDataset(dataroot, train=True, augment=augments)

    else:
        train_dataset = ClassificationDataset(dataroot, train=True, augment=augments)
        test_dataset = ClassificationDataset(dataroot, train=False)
        print(len(test_dataset))
        test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                           shuffle=True, pin_memory=True)
    print(len(train_dataset))
    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)

    # ========== Network ==========
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Mesh_mae(masking_ratio=args.mask_ratio,
                   channels=args.channels,
                   num_heads=args.heads,
                   encoder_depth=args.encoder_depth,
                   embed_dim=args.dim,
                   decoder_num_heads=args.decoder_num_heads,
                   decoder_depth=args.decoder_depth,
                   decoder_embed_dim=args.decoder_dim,
                   patch_size=args.patch_size,
                   weight=args.weight
                   ).to(device)

    # ========== Optimizer ==========
    if args.optim.lower() == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones.lower() != 'none':
        ms = args.lr_milestones
        ms = ms.split()
        ms = [int(j) for j in ms]
        scheduler = MultiStepLR(optim, milestones=ms, gamma=0.1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=int(args.num_warmup_steps),
                                                    num_training_steps=args.max_epoch + 1)

    print(scheduler)

    # ========== MISC ==========

    checkpoint_names = []
    checkpoint_path = os.path.join('checkpoints', args.name)
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint.lower() != 'none':
        net.load_state_dict(torch.load(args.checkpoint), strict=True)


    train.best_loss = 999
    train.best_epoch = 0
    # ========== Start Training ==========

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            print('epoch', epoch)
            train(net, optim, scheduler, args.name, train_data_loader, epoch, args)
            print('train finished')



    else:
        test(net,  args.name, test_data_loader, 0, args)
