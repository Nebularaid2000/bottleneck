import os
import random
import torch
import argparse
import numpy as np

from io_handler import PairIoHandler, PlayerIoHandler, SampleIoHandler, set_args
from util.utils import clamp, seed_torch, prepare, log_args_and_backup_code


def gen_pairs(grid_size: int, pair_num: int, stride: int = 1) -> np.ndarray:
    """
    Input:
        grid_size: int, the image is partitioned to grid_size * grid_size patches. Each patch is considered as a player.
        pair_num: int, how many (i,j) pairs to sample for one image
        stride: int, j should be sampled in a neighborhood of i. stride is the radius of the neighborhood.
            e.g. if stride=1, then j should be sampled from the 8 neighbors around i
                if stride=2, then j should be sampled from the 24 neighbors around i
    Return:
        total_pairs: (pair_num,2) array, sampled (i,j) pairs
    """

    neighbors = [(i, j) for i in range(-stride, stride + 1)
                 for j in range(-stride, stride + 1)
                 if
                 i != 0 or j != 0]

    total_pairs = []
    for _ in range(pair_num):
        while True:
            x1 = np.random.randint(0, grid_size)
            y1 = np.random.randint(0, grid_size)
            point1 = x1 * grid_size + y1

            neighbor = random.choice(neighbors)
            x2 = clamp(x1 + neighbor[0], 0, grid_size - 1)
            y2 = clamp(y1 + neighbor[1], 0, grid_size - 1)
            point2 = x2 * grid_size + y2

            if point1 == point2:
                continue

            if [point1, point2] in total_pairs or [point2, point1] in total_pairs:
                continue
            else:
                total_pairs.append(list([point1, point2]))
                break

    return np.array(total_pairs)


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
    parser.add_argument('--gpu_id', default=1, type=int, help="GPU ID")
    parser.add_argument('--chosen_class', default='random', type=str, choices=['random'])
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

    sample_io_handler = SampleIoHandler(args)
    pair_io_handler = PairIoHandler(args)
    player_io_handler = PlayerIoHandler(args)

    image_list_selected = sample_io_handler.load()
    total_pairs = []
    model, dataloader = prepare(args, train=True)

    # sample (i,j) pairs and contexts S
    for index, (name, _, _) in enumerate(dataloader):
        print('\rPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(image_list_selected)), end='')

        seed_torch(1000 * index + args.seed) # seed for sampling (i,j) pair
        pairs = gen_pairs(args.grid_size, args.pairs_number, args.stride)
        for ratio in args.ratios:
            m = int((args.grid_size ** 2 - 2) * ratio)  # m-order

            seed_torch(1000 * index + m + 1 + args.seed) # seed for sampling context S
            players_with_ratio = []
            for pair in pairs:
                point1, point2 = pair[0], pair[1]
                context = list(range(args.grid_size ** 2))
                context.remove(point1)
                context.remove(point2)

                curr_players = []
                for _ in range(args.samples_number_of_s):
                    curr_players.append(np.random.choice(context, m, replace=False)) # sample contexts of cardinality m

                players_with_ratio.append(curr_players)
            players_with_ratio = np.array(players_with_ratio)  # (pair_num, sample_num_of_s, m), contexts S of cardinality m for different (i,j) pairs
            print(players_with_ratio.shape)
            player_io_handler.save(round(ratio * 100), name[0], players_with_ratio)
        total_pairs.append(pairs)
    total_pairs = np.array(total_pairs)  # (num_imgs, num_pairs, 2), all (i,j) pairs
    print(total_pairs.shape)
    pair_io_handler.save(total_pairs)
    print('\nDone!')
