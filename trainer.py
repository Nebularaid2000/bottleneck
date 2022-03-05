import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from util.utils import seed_torch, mkdir
import datetime
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from util.train_util import (prepare_model, prepare_dataset, log_hparam_and_backup_code, save_checkpoint, AverageMeter, accuracy)


parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='our_alexnet',
                    choices=['our_alexnet'])

parser.add_argument('--dataset', default='cifar10', choices=['cifar10'])

parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--scheduler', default='log', type=str, choices=['log','step'])
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--logspace',default=1, type=int,
                    help='log space of learning rate decay')


parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation/training set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='checkpoints', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('--gpu_id', help='single gpu id to use',
                    type=int, default=1)
parser.add_argument('--seed', help='random seed',
                    type=int, default=0)
parser.add_argument('--horizontal_flip', default=True, type=bool)
parser.add_argument('--random_crop', default=True, type=bool)


def main():
    best_prec1 = 0
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.arch == "our_alexnet":
        args.lr = 0.01
        args.weight_decay = 5e-4

    # set save path
    if args.scheduler == "log":
        dp_prefix = f"normal_lr{args.lr}_log{args.logspace}"
    elif args.scheduler == "step":
        dp_prefix = f"normal_lr{args.lr}_step{args.milestones}_gamma{args.gamma}"
    else:
        raise Exception(f"scheduler [{args.scheduler}] not implemented")
    args.save_subdir = dp_prefix + "_da"
    if args.horizontal_flip:
        args.save_subdir += "_flip"
    if args.random_crop:
        args.save_subdir += "_crop"
    args.save_subdir = args.save_subdir + f"_seed{args.seed}"
    args.result_path = os.path.join(args.save_dir, "%s_%s" % (args.arch,args.dataset), args.save_subdir)

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


    if args.scheduler == "log":
        all_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epochs)
    elif args.scheduler == "step":
        lr, last = args.lr, 0
        all_lr = []
        for i in range(len(args.milestones)):
            all_lr.extend([lr for _ in range(args.milestones[i] - last)])
            last = args.milestones[i]
            lr *= args.gamma
        all_lr.extend([lr for _ in range(args.epochs - last)])
        all_lr = np.array(all_lr)
    else:
        raise Exception(f"scheduler [{args.scheduler}] not implemented")

    train_acc_list, train_loss_list, val_acc_list, val_loss_list = [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        seed_torch(epoch + args.batch_size)
        # adjusr lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = all_lr[epoch]

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_loss, train_prec1 = train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, val_prec1 = validate(args, val_loader, model, criterion, epoch)

        # draw loss and top1 acc curve
        train_loss_list.append(train_loss), val_loss_list.append(val_loss)
        train_acc_list.append(train_prec1), val_acc_list.append(val_prec1)
        draw_acc_loss_figure(os.path.join(args.result_path, "loss_acc.png"), train_acc_list, val_acc_list, train_loss_list, val_loss_list)

        if val_prec1 >= best_prec1:
            best_prec1 = val_prec1
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
            save_list(os.path.join(args.result_path, 'training_stats.pth'), train_acc_list, val_acc_list, train_loss_list, val_loss_list)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, filename=os.path.join(args.result_path, 'model_last.pth'))

    save_list(os.path.join(args.result_path, 'training_stats.pth'), train_acc_list, val_acc_list, train_loss_list,val_loss_list)


def train(args, train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(args.device)
        input = input.to(args.device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0] # can measure different topk acc, but here we only measure top1 acc
        losses.update(loss.item(), input.size(0)) # input.size(0) is the batch size
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' # loss of current batch (avg loss across all seen batches)
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return losses.avg, top1.avg # running loss, running acc

def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(args.device)
            input = input.to(args.device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return losses.avg, top1.avg  # running loss, running acc

def save_list(path, train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    training_stats = {
        'train_acc': train_acc_list,
        'val_acc': val_acc_list,
        'train_loss': train_loss_list,
        'val_loss': val_loss_list
    }
    torch.save(training_stats, path)

def draw_acc_loss_figure(path, train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    font_size = 16
    x = range(len(train_loss_list))
    y1 = train_acc_list
    y2 = val_acc_list
    y3 = train_loss_list
    y4 = val_loss_list

    plt.figure(figsize=(8,7))
    acc_figure = plt.subplot(211)
    plt.plot(x, y1, color="blue", label="train acc")
    plt.plot(x, y2, color="blue", linestyle="dashed", label="val acc")
    plt.xlabel("epoch", fontsize=font_size)
    plt.legend()
    plt.tick_params(labelsize=font_size)
    acc_figure.set_title("acc",fontsize=font_size)
    loss_figure = plt.subplot(212)
    plt.plot(x, y3, color="blue", label="train loss")
    plt.plot(x, y4, color="blue", linestyle="dashed", label="val loss")
    loss_figure.set_title("loss",fontsize=font_size)
    plt.xlabel("epoch", fontsize=font_size)
    plt.legend()
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



if __name__ == '__main__':
    main()
