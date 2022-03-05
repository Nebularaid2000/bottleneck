import os
from typing import Iterable, Tuple, Union
import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import argparse

from io_handler import InteractionIoHandler, SampleIoHandler, set_args

font_size = 20


def get_name_list(args, sample_io_handler: SampleIoHandler):
    if args.dataset == "cifar10":
        names = list(map(lambda item: item[0] + "_%05d" % item[1], sample_io_handler.load()))
    else:
        raise Exception("Dataset not implemented")
    return names


def get_interactions_strength(args, sample_io_handler: SampleIoHandler, interaction_io_handler: InteractionIoHandler, ratios: Iterable[int]) -> np.ndarray:
    names = get_name_list(args, sample_io_handler)
    interactions_list = []
    for ratio in ratios:
        interactions_list_with_ratio = []
        for name in names:
            interactions = interaction_io_handler.load(round(ratio * 100), name) # (pair_num, sample_num_s)
            interactions = np.mean(interactions, axis=1) # (pair_num)
            interactions_list_with_ratio.append(interactions)
        interactions_list.append(interactions_list_with_ratio) # append (img_num, pair_num)
    interactions_list = np.array(interactions_list) # (ratio_num, img_num, pair_num)
    interactions_list_abs = np.abs(interactions_list)  # (ratio_num, img_num, pair_num) of |I_ij^(m)|
    Im = interactions_list_abs.mean(axis=(1,2)) # (ratio_num,)
    Jm = Im / Im.mean() # (ratio_num,)
    return Jm


def draw_curve(interactions: Iterable[float], ratios: Iterable[float], title: str, filepath: str, ylim1=None, ylim2=None):
    plt.plot(ratios, interactions)
    plt.title(title, fontsize=font_size)
    plt.xlabel('Order', fontsize=font_size)
    plt.ylabel('Interaction Strength', fontsize=font_size)
    plt.xticks(np.arange(0, 1.01, 0.2), map(lambda r: '%.1fn' % r, np.arange(0, 1.01, 0.2)))
    plt.ylim(ylim1, ylim2)
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    for format in ['png']:
        plt.savefig(f'{filepath}.{format}', format=format)
    plt.close()



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
    parser.add_argument('--softmax_type', default='modified', type=str, choices=['normal', 'modified','yi'],
                        help="reward function for interaction")
    parser.add_argument('--out_type', default='gt', type=str, choices=['gt'])
    parser.add_argument('--chosen_class', default='random', type=str, choices=['random'])
    parser.add_argument('--gpu_id', default=1, type=int, help="GPU ID")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int)

    args = parser.parse_args()

    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    FIGURES_DIR = os.path.join(args.prefix, "results", args.output_dirname, "MODEL_%s_DATA_%s"%(args.arch, args.dataset),
                               args.figures_dirname + "_out_%s_softmax_%s"%(args.out_type, args.softmax_type))
    if not os.path.isdir(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    sample_io_handler = SampleIoHandler(args)
    interaction_io_handler = InteractionIoHandler(args)

    Jm = get_interactions_strength(args, sample_io_handler, interaction_io_handler, args.ratios) # (ratio_num, img_num, pair_num) of I_ij^(m)

    print(Jm)
    draw_curve(Jm, args.ratios, 'Interaction Strength of Order (average)',
               os.path.join(FIGURES_DIR, 'Interaction Strength of Order (average)'), ylim1=0)

