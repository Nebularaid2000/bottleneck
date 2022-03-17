import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import argparse
import os
import time
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from io_handler import (InteractionLogitIoHandler, PairIoHandler,
                        PlayerIoHandler, set_args)
from util.utils import prepare, seed_torch, normalize, log_args_and_backup_code, mkdir

MAX_BS = 4 * 100


def compute_order_interaction_img(args, model: torch.nn.Module, feature: torch.Tensor, feature_shape,
                                  name: str, pairs: np.ndarray,
                                  ratio: float, player_io_handler: PlayerIoHandler,
                                  interaction_logit_io_handler: InteractionLogitIoHandler):
    """
    Input:
        args: args
        model: nn.Module, model to be evaluated
        feature: (1,C,H,W) tensor
        feature_shape: tuple, shape of feature
        name: str, name of this sample
        pairs: (pairs_num, 2) array, (i,j) pairs
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        player_io_handler:
        interaction_logit_io_handler:
    Return:
        None
    """
    time0 = time.time()
    model.to(args.device)
    order = int((args.grid_size ** 2 - 2) * ratio)
    print("m=%d" % order)

    with torch.no_grad():
        model.eval()
        channels = feature.size(1)
        players = player_io_handler.load(round(ratio * 100), name)
        ori_logits = []

        forward_mask = []
        for index, pair in enumerate(pairs):
            print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(pairs)), end='')
            point1, point2 = pair[0], pair[1]

            players_curr_pair = players[index] # context S for this pair of (i,j)
            mask = torch.zeros((4 * args.samples_number_of_s, channels, args.grid_size ** 2), device=args.device)

            if order != 0: # if order == 0, then S=emptyset, we don't need to set S
                S_cardinality = players_curr_pair.shape[1]  # |S|
                assert S_cardinality == order
                idx_multiple_of_4 = 4 * np.arange(args.samples_number_of_s)  # indices: 0, 4, 8...
                stack_idx = np.stack([idx_multiple_of_4] * S_cardinality, axis=1)  # stack the indices to match the shape of player_curr_i
                mask[stack_idx, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+1, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+2, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)
                mask[stack_idx+3, :, players_curr_pair] = 1  # set S for v(S U {i}) and v(S)

            mask[4 * np.arange(args.samples_number_of_s) + 1, :, point1] = 1  # S U {i}
            mask[4 * np.arange(args.samples_number_of_s) + 2, :, point2] = 1  # S U {j}
            mask[4 * np.arange(args.samples_number_of_s), :, point1] = 1  # S U {i,j}
            mask[4 * np.arange(args.samples_number_of_s), :, point2] = 1  # S U {i,j}

            mask = mask.view(4 * args.samples_number_of_s, channels, args.grid_size, args.grid_size)
            mask = F.interpolate(mask.clone(), size=[feature_shape[2], feature_shape[3]], mode='nearest').float()


            if len(mask) > MAX_BS: # if sample number of S is too large (especially for vgg19), we need to split one batch into several iterations
                iterations = math.ceil(len(mask) / MAX_BS)
                for it in range(iterations): # in each iteration, we compute output for MAX_BS images
                    batch_mask = mask[it * MAX_BS : min((it+1) * MAX_BS, len(mask))]
                    expand_feature = feature.expand(len(batch_mask), channels, feature_shape[2], feature_shape[3]).clone()
                    masked_feature = batch_mask * expand_feature

                    output_ori = model(masked_feature)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'
                    ori_logits.append(output_ori.detach())

            else: # if sample number of S is small, we can concatenate several batches and do a single inference
                forward_mask.append(mask)
                if (len(forward_mask) < args.cal_batch // args.samples_number_of_s) and (index < args.pairs_number - 1):
                    continue
                else:
                    forward_batch = len(forward_mask) * args.samples_number_of_s
                    batch_mask = torch.cat(forward_mask, dim=0)
                    expand_feature = feature.expand(4 * forward_batch, channels, feature_shape[2], feature_shape[3]).clone()
                    masked_feature = batch_mask * expand_feature

                    output_ori = model(masked_feature)
                    assert not torch.isnan(output_ori).any(), 'there are some nan numbers in the model output'

                    ori_logits.append(output_ori.detach())
                    forward_mask = []
        print('done time: ', time.time() - time0)

        all_logits = torch.cat(ori_logits, dim=0)  # (pairs_num*4*samples_number_of_s, class_num)
        print("all_logits shape: ", all_logits.shape)
        interaction_logit_io_handler.save(round(ratio * 100), name, all_logits)


def compute_interactions(args, model: nn.Module, dataloader: DataLoader, pair_io_handler: PairIoHandler,
                         player_io_handler: PlayerIoHandler, interaction_logit_io_handler: InteractionLogitIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        total_pairs = pair_io_handler.load()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))

            image = image.to(args.device)
            label = label.to(args.device)

            image = normalize(args, image)
            pairs = total_pairs[index]

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                compute_order_interaction_img(args, model, image, image.shape, name[0], pairs, ratio, player_io_handler,
                                              interaction_logit_io_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirname', default="result", type=str)
    parser.add_argument('--inter_type', default="pixel", type=str, choices=["pixel"])
    parser.add_argument('--arch', default="our_alexnet_cifar10_normal_lr0.01_log1_da_flip_crop_best", type=str,
                        choices=[
                            # --- cifar 10 ---
                            "our_alexnet_cifar10_normal_lr0.01_log1_da_flip_crop_best",

                            "our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_0.5_0.0_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best",
                            "our_alexnet_cifar10_dp_pos0_deltav_baseline_0.7_0.3_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best",
                            "our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_1.0_0.7_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best",

                        ])
    parser.add_argument("--dataset", default="cifar10", type=str, choices=['cifar10'])
    parser.add_argument("--cal_batch", default=100, type=int, help='calculate # of images per batch')
    parser.add_argument('--gpu_id', default=1, type=int, help="GPU ID")
    parser.add_argument('--chosen_class', default='random', type=str,choices=['random'])
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int,
                        help="partition the input image to grid_size * grid_size patches"
                             "each patch is considered as a player")

    args = parser.parse_args()

    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_torch(args.seed)

    log_args_and_backup_code(args, __file__)

    pair_io_handler = PairIoHandler(args)
    player_io_handler = PlayerIoHandler(args)
    interaction_logit_io_handler = InteractionLogitIoHandler(args)

    model, dataloader = prepare(args, train=True)
    compute_interactions(args, model, dataloader, pair_io_handler, player_io_handler, interaction_logit_io_handler)
