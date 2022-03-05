import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.alexnet import our_alexnet

from util.utils import LogWriter, CIFAR10_MEAN, CIFAR10_STD
import socket
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def log_hparam_and_backup_code(args, file_path):
    file_name = os.path.basename(file_path)
    logfile = LogWriter(os.path.join(args.result_path, "hparam.txt"))
    for k, v in args.__dict__.items():
        logfile.cprint(f"{k} : {v}")
    logfile.cprint("Numpy: {}".format(np.__version__))
    logfile.cprint("Pytorch: {}".format(torch.__version__))
    logfile.cprint("torchvision: {}".format(torchvision.__version__))
    logfile.cprint("Cuda: {}".format(torch.version.cuda))
    logfile.cprint("hostname: {}".format(socket.gethostname()))
    logfile.close()

    os.system(f'cp {file_path} {args.result_path}/{file_name}.backup')

def prepare_model(args):
    if args.arch == "our_alexnet":
        model = our_alexnet(dataset=args.dataset)
    else:
        raise Exception("Model not implemented")
    model.to(args.device)
    print(str(model))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            if 'best_prec1' in checkpoint:
                best_prec1 = checkpoint['best_prec1']
                print("best acc", best_prec1)
            if not ('state_dict' in checkpoint):
                sd = checkpoint
            else:
                sd = checkpoint['state_dict']
            # load with models trained on a single gpu or multiple gpus
            if 'module.' in list(sd.keys())[0]:
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model


def prepare_dataset(args):
    if args.dataset == "cifar10":
        normalize = transforms.Normalize(mean=CIFAR10_MEAN,
                                         std=CIFAR10_STD)

        train_transform_list = []

        if args.horizontal_flip:
            train_transform_list.append(transforms.RandomHorizontalFlip())
        if args.random_crop:
            train_transform_list.append(transforms.RandomCrop(32, 4))

        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(normalize)
        print("train transform list: ", train_transform_list)
        train_transform = transforms.Compose(train_transform_list)

        train_set = datasets.CIFAR10(root='./datasets', train=True, transform=train_transform, download=True)
        val_set = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    else:
        raise Exception("dataset [%s] not implemented" % args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    return train_loader, val_loader


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res