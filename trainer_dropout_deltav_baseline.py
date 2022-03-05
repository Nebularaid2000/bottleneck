import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from util.utils import seed_torch, mkdir
import datetime
import torchvision
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from util.train_util import (prepare_model, prepare_dataset, log_hparam_and_backup_code, save_checkpoint, AverageMeter, accuracy)

parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='our_alexnet',
                    choices=['our_alexnet'])

parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--logspace',default=1, type=int,
                    help='log space of learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation/training set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='checkpoints', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=50)
parser.add_argument('--gpu_id', help='single gpu id to use',
                    type=int, default=1)
parser.add_argument('--seed', help='random seed',
                    type=int, default=0)
parser.add_argument('--dropout_pos', help='which layer to apply dropout, 0: dropout input image(pixel-wise)',
                    type=int, default=0, choices=[0])
parser.add_argument('--horizontal_flip', default=True, type=bool)
parser.add_argument('--random_crop', default=True, type=bool)
parser.add_argument('--preserve_rate1', type=float, default=0.7, help="r2 in paper")
parser.add_argument('--preserve_rate2', type=float, default=0.3, help="r1 in paper")
parser.add_argument('--lam1', type=float, default=1.0, help="coefficient of ori loss")
parser.add_argument('--lam2', type=float, default=1.0, help="coefficient of deltav loss")
parser.add_argument('--grid_size', type=int, default=16, help="number of grids for dropout, default:16(16x16 grids to dropout)")
parser.add_argument('--S_sample_num', type=int, default=1, help="number of (S1, S2) to sample", choices=[1])


def main():
    best_prec1 = 0
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.arch == "our_alexnet":
        args.lr = 0.01
        args.weight_decay = 5e-4

    # set save path
    dp_prefix = "dp_pos%d_" % (args.dropout_pos)
    args.save_subdir = dp_prefix + f"deltav_baseline_{args.preserve_rate1}_{args.preserve_rate2}_lam{args.lam1}_{args.lam2}_grid{args.grid_size}_lr{args.lr}_log{args.logspace}_da"
    if args.horizontal_flip:
        args.save_subdir += "_flip"
    if args.random_crop:
        args.save_subdir += "_crop"
    args.save_subdir = args.save_subdir + f"_seed{args.seed}"
    args.result_path = os.path.join(args.save_dir, "%s_%s" % (args.arch, args.dataset), args.save_subdir)

    if args.preserve_rate2 != 0:
        args.align_coef = args.preserve_rate1 / args.preserve_rate2
    else:  # if preserve_rate2=0, we just set the align coefficient to 1
        args.align_coef = 1

    seed_torch(args.seed)

    mkdir(args.result_path)

    log_hparam_and_backup_code(args, __file__)
    model = prepare_model(args)
    train_loader, val_loader = prepare_dataset(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)

    train_total_loss_list, val_total_loss_list = [], []
    train_ori_loss_list, train_deltav_loss_list, val_ori_loss_list, val_deltav_loss_list = [], [],[],[]
    train_ori_acc_list, train_deltav_acc_list, val_ori_acc_list, val_deltav_acc_list = [], [],[],[]

    for epoch in range(args.start_epoch, args.epochs):
        seed_torch(epoch + args.batch_size)
        # adjusr lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = logspace_lr[epoch]

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_total_loss, train_ori_loss, train_deltav_loss, train_ori_acc, train_deltav_acc = train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_total_loss, val_ori_loss, val_deltav_loss, val_ori_acc, val_deltav_acc = validate(args, val_loader, model, criterion, epoch)

        # draw loss and top1 acc curve
        train_total_loss_list.append(train_total_loss), val_total_loss_list.append(val_total_loss)

        train_ori_loss_list.append(train_ori_loss), val_ori_loss_list.append(val_ori_loss)
        train_deltav_loss_list.append(train_deltav_loss), val_deltav_loss_list.append(val_deltav_loss)

        train_ori_acc_list.append(train_ori_acc), val_ori_acc_list.append(val_ori_acc)
        train_deltav_acc_list.append(train_deltav_acc), val_deltav_acc_list.append(val_deltav_acc)

        draw_acc_loss_figure(args, os.path.join(args.result_path, "loss_acc.png"),
                             train_total_loss_list, val_total_loss_list,
                             train_ori_loss_list, val_ori_loss_list,
                             train_deltav_loss_list, val_deltav_loss_list,
                             train_ori_acc_list, val_ori_acc_list,
                             train_deltav_acc_list, val_deltav_acc_list)

        if val_ori_acc >= best_prec1:
            best_prec1 = val_ori_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.result_path, 'model_best.pth'))

        if epoch > 0 and (epoch+1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.result_path, 'checkpoint_%d.pth' % epoch))
            save_list(os.path.join(args.result_path, 'training_stats.pth'),
                      train_total_loss_list, val_total_loss_list,
                      train_ori_loss_list, val_ori_loss_list,
                      train_deltav_loss_list, val_deltav_loss_list,
                      train_ori_acc_list, val_ori_acc_list,
                      train_deltav_acc_list, val_deltav_acc_list)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, filename=os.path.join(args.result_path, 'model_last.pth'))

    save_list(os.path.join(args.result_path, 'training_stats.pth'),
              train_total_loss_list, val_total_loss_list,
              train_ori_loss_list, val_ori_loss_list,
              train_deltav_loss_list, val_deltav_loss_list,
              train_ori_acc_list, val_ori_acc_list,
              train_deltav_acc_list, val_deltav_acc_list)

def get_baseline_value(args, input, model):
    """
    Input
        args: args
        input: (N,3,H,W) tensor, input image (in batch), always H=W=224
        model: nn.Module, model to be evaluated
    Return:
        baseline: (N, C) tensor, baseline for each channel of a feature map at args.dropout_pos
    """
    with torch.no_grad():
        gray_image = torch.zeros_like(input)
        feature_map = model.all_layers[:args.dropout_pos](gray_image)  # (N,C,H',W') feature map at a specific layer corresponding to the gray image
        baseline = torch.mean(feature_map, dim=(2, 3))  # (N,C)
    return baseline


def get_dropout_output(args, x_inter, baseline):
    """
    Input
        args: args
        x_inter: (N,C,H',W') tensor, feature map to apply dropout
        model: nn.Module, model to be evaluated
        baseline: (N, C) tensor
    Return:
        x_dp1: (N,C,H',W') tensor
        x_dp2: (N,C,H',W') tensor
    """
    N, C, H, W = x_inter.size()
    num_grids = args.grid_size ** 2
    preserve_number1 = int(args.preserve_rate1 * num_grids)  # preserve_rate1*H*W
    preserve_number2 = int(args.preserve_rate2 * num_grids)  # preserve_rate2*H*W
    random_perms = []
    for k in range(N):  # for all imgs, generate a permutation of the feature map pixels
        random_perms.append(np.random.permutation(num_grids))
    random_perms = np.stack(random_perms, axis=0)  # (N,H*W)
    index_mask1 = random_perms[:, preserve_number1:]
    index_mask2 = random_perms[:, preserve_number2:]

    idx = np.arange(N)  # indices: 0, 1, 2...

    mask1 = torch.ones(N, 1, num_grids, device=args.device)
    if index_mask1.shape[1] != 0: # else, we will preserve all elements in x_inter, since mask1 is a all-one mask
        stack_idx1 = np.stack([idx] * index_mask1.shape[1], axis=1)
        mask1[stack_idx1, :, index_mask1] = 0  # only contains 0 and 1
    mask1 = mask1.reshape((N, 1, args.grid_size, args.grid_size))
    mask1 = F.interpolate(mask1.clone(), size=[H, W], mode='nearest').float()
    x_dp1 = x_inter * mask1

    baseline_mask1 = torch.zeros(N, C, num_grids, device=args.device)
    if index_mask1.shape[1] != 0:
        baseline_mask1[stack_idx1, :, index_mask1] = baseline[:, None, :]
    baseline_mask1 = baseline_mask1.reshape((N, C, args.grid_size, args.grid_size))
    baseline_mask1 = F.interpolate(baseline_mask1.clone(), size=[H, W], mode='nearest').float()
    x_dp1 = x_dp1 + baseline_mask1


    mask2 = torch.ones(N, 1, num_grids, device=args.device)
    if index_mask2.shape[1] != 0:  # else, we will preserve all elements in x_inter, since mask2 is a all-one mask
        stack_idx2 = np.stack([idx] * index_mask2.shape[1], axis=1)
        mask2[stack_idx2, :, index_mask2] = 0
    mask2 = mask2.reshape((N, 1, args.grid_size, args.grid_size))
    mask2 = F.interpolate(mask2.clone(), size=[H, W], mode='nearest').float()
    x_dp2 = x_inter * mask2

    baseline_mask2 = torch.zeros(N, C, num_grids, device=args.device)
    if index_mask2.shape[1] != 0:
        baseline_mask2[stack_idx2, :, index_mask2] = baseline[:,None,:]
    baseline_mask2 = baseline_mask2.reshape((N, C, args.grid_size, args.grid_size))
    baseline_mask2 = F.interpolate(baseline_mask2.clone(), size=[H, W], mode='nearest').float()
    x_dp2 = x_dp2 + baseline_mask2

    return x_dp1, x_dp2


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ori_losses = AverageMeter()
    deltav_losses = AverageMeter()

    ori_accs = AverageMeter()
    deltav_accs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(args.device)
        input = input.to(args.device)

        output = model.all_layers(input)  # original output

        # backward twice to save memory
        # first calculate original loss and backward
        ori_loss = criterion(output, target) * args.lam1
        optimizer.zero_grad()
        ori_loss.backward()

        x_inter = model.all_layers[:args.dropout_pos](input)  # (N,C,H,W)
        baseline = get_baseline_value(args, input, model) # (N, C)

        delta_v = 0.
        for S_idx in range(args.S_sample_num): # default is 1
            x_dp1, x_dp2 = get_dropout_output(args, x_inter, baseline)
            output1 = model.all_layers[args.dropout_pos:](x_dp1)
            output2 = model.all_layers[args.dropout_pos:](x_dp2)
            delta_v = delta_v + output1 - output2 * args.align_coef # delta v = v(0.7 n) - alpha*v(0.3n)
        delta_v = delta_v / args.S_sample_num

        # backward twice to save memory (continued)
        # calculate delta v loss and backward again
        deltav_loss = criterion(delta_v, target) * args.lam2
        deltav_loss.backward()
        optimizer.step()
        loss = ori_loss.item() + deltav_loss.item()

        # measure accuracy and record loss
        ori_acc = accuracy(output.data, target)[0]
        deltav_acc = accuracy(delta_v.data, target)[0]

        losses.update(loss, input.size(0)) # input.size(0) is the batch size
        ori_losses.update(ori_loss.item(), input.size(0)) # input.size(0) is the batch size
        deltav_losses.update(deltav_loss.item(), input.size(0)) # input.size(0) is the batch size

        ori_accs.update(ori_acc.item(), input.size(0))
        deltav_accs.update(deltav_acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'total loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'ori loss {ori_losses.val:.4f} ({ori_losses.avg:.4f})\t' # loss of current batch (avg loss across all seen batches)
                  'deltav loss {deltav_losses.val:.4f} ({deltav_losses.avg:.4f})\t'
                  'ori acc {ori_accs.val:.3f} ({ori_accs.avg:.3f})\t'
                  'deltav acc {deltav_accs.val:.3f} ({deltav_accs.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      losses=losses, ori_losses=ori_losses, deltav_losses=deltav_losses,
                      ori_accs=ori_accs, deltav_accs=deltav_accs))
    return losses.avg, ori_losses.avg, deltav_losses.avg, ori_accs.avg, deltav_accs.avg

def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    ori_losses = AverageMeter()
    deltav_losses = AverageMeter()

    ori_accs = AverageMeter()
    deltav_accs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(args.device)
            input = input.to(args.device)

            # compute output
            output = model.all_layers(input)
            ori_loss = criterion(output, target) * args.lam1

            x_inter = model.all_layers[:args.dropout_pos](input)  # (N,C,H,W)
            baseline = get_baseline_value(args, input, model) # (N, C)

            delta_v = 0.
            for S_idx in range(args.S_sample_num):  # default is 1
                x_dp1, x_dp2 = get_dropout_output(args, x_inter, baseline)
                output1 = model.all_layers[args.dropout_pos:](x_dp1)
                output2 = model.all_layers[args.dropout_pos:](x_dp2)
                delta_v = delta_v + output1 - output2 * args.align_coef  # delta v = v(0.7n) - alpha*v(0.3n)
            delta_v = delta_v / args.S_sample_num

            deltav_loss = criterion(delta_v, target) * args.lam2
            loss = ori_loss.item() + deltav_loss.item()

            # measure accuracy and record loss
            ori_acc = accuracy(output.data, target)[0]
            deltav_acc = accuracy(delta_v.data, target)[0]

            losses.update(loss, input.size(0))  # input.size(0) is the batch size
            ori_losses.update(ori_loss.item(), input.size(0))  # input.size(0) is the batch size
            deltav_losses.update(deltav_loss.item(), input.size(0))  # input.size(0) is the batch size

            ori_accs.update(ori_acc.item(), input.size(0))
            deltav_accs.update(deltav_acc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'total loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      'ori loss {ori_losses.val:.4f} ({ori_losses.avg:.4f})\t'  # loss of current batch (avg loss across all seen batches)
                      'deltav loss {deltav_losses.val:.4f} ({deltav_losses.avg:.4f})\t'
                      'ori acc {ori_accs.val:.3f} ({ori_accs.avg:.3f})\t'
                      'deltav acc {deltav_accs.val:.3f} ({deltav_accs.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    losses=losses, ori_losses=ori_losses, deltav_losses=deltav_losses,
                    ori_accs=ori_accs, deltav_accs=deltav_accs))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=ori_accs))

    return losses.avg, ori_losses.avg, deltav_losses.avg, ori_accs.avg, deltav_accs.avg


def save_list(path, train_total_loss_list, val_total_loss_list,
              train_ori_loss_list, val_ori_loss_list,
              train_deltav_loss_list, val_deltav_loss_list,
              train_ori_acc_list, val_ori_acc_list,
              train_deltav_acc_list, val_deltav_acc_list):
    training_stats = {
        'train_total_loss': train_total_loss_list,
        'val_total_loss': val_total_loss_list,
        'train_ori_loss': train_ori_loss_list,
        'val_ori_loss': val_ori_loss_list,
        'train_deltav_loss': train_deltav_loss_list,
        'val_deltav_loss': val_deltav_loss_list,
        'train_ori_acc': train_ori_acc_list,
        'val_ori_acc': val_ori_acc_list,
        'train_deltav_acc': train_deltav_acc_list,
        'val_deltav_acc': val_deltav_acc_list,
    }
    torch.save(training_stats, path)

def draw_acc_loss_figure(args, path, train_total_loss_list, val_total_loss_list,
              train_ori_loss_list, val_ori_loss_list,
              train_deltav_loss_list, val_deltav_loss_list,
              train_ori_acc_list, val_ori_acc_list,
              train_deltav_acc_list, val_deltav_acc_list):
    font_size = 16
    x = range(len(train_total_loss_list))

    plt.figure(figsize=(8,7))
    acc_figure = plt.subplot(211)
    plt.plot(x, train_ori_acc_list, color="green", label="train ori acc")
    plt.plot(x, val_ori_acc_list, color="green", linestyle="dashed",label="val ori acc")
    plt.plot(x, train_deltav_acc_list, color="red", label="train deltav acc")
    plt.plot(x, val_deltav_acc_list, color="red", linestyle="dashed",label="val deltav acc")
    plt.xlabel("epoch", fontsize=font_size)
    plt.legend()
    plt.tick_params(labelsize=font_size)
    acc_figure.set_title("acc",fontsize=font_size)

    loss_figure = plt.subplot(212)
    plt.plot(x, train_total_loss_list, color="blue", label="train total loss")
    plt.plot(x, val_total_loss_list, color="blue", linestyle="dashed", label="val total loss")
    plt.plot(x, train_ori_loss_list, color="green", label="train ori loss")
    plt.plot(x, val_ori_loss_list, color="green", linestyle="dashed", label="val ori loss")
    plt.plot(x, train_deltav_loss_list, color="red", label="train deltav loss")
    plt.plot(x, val_deltav_loss_list, color="red", linestyle="dashed", label="val deltav loss")
    loss_figure.set_title("loss",fontsize=font_size)
    plt.xlabel("epoch", fontsize=font_size)
    plt.legend()
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
