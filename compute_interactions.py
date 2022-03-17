import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import os
import argparse

from io_handler import InteractionIoHandler, InteractionLogitIoHandler, set_args
from util.utils import prepare, seed_torch, get_reward, log_args_and_backup_code


def compute_order_interaction_img(args, name: str, label: torch.Tensor, ratio: float,
                                  interaction_logit_io_handler: InteractionLogitIoHandler,
                                  interaction_io_handler: InteractionIoHandler):
    """
    Input:
        args: args
        name: str, name of this sample
        label: (1,) tensor, label of this sample
        ratio: float, ratio of the order of the interaction, order=(n-2)*ratio
        interaction_logit_io_handler:
        interaction_io_handler:
    Return:
        None
    """
    interactions = []

    logits = interaction_logit_io_handler.load(round(ratio * 100), name)
    logits = logits.reshape((args.pairs_number, args.samples_number_of_s * 4, args.class_number)) # load saved logits

    for index in range(args.pairs_number):
        print('\r\t\tPairs: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, args.pairs_number), end='')
        output_ori = logits[index, :, :]

        v = get_reward(args, output_ori, label)  # (4*samples_number_of_s,)

        # Delta v(i,j,S) = v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)
        score_ori = v[4 * np.arange(args.samples_number_of_s)] + v[4 * np.arange(args.samples_number_of_s) + 3] \
                    - v[4 * np.arange(args.samples_number_of_s) + 1] - v[4 * np.arange(args.samples_number_of_s) + 2]
        interactions.extend(score_ori.tolist())

    print('')
    interactions = np.array(interactions).reshape(-1, args.samples_number_of_s) # (pair_num, sample_num)
    assert interactions.shape[0] == args.pairs_number

    interaction_io_handler.save(round(ratio * 100), name, interactions)  # (pair_num, sample_num)


def compute_interactions(args, model: nn.Module, dataloader: DataLoader, interaction_logit_io_handler: InteractionLogitIoHandler, interaction_io_handler: InteractionIoHandler):
    model.to(args.device)

    with torch.no_grad():
        model.eval()
        for index, (name, image, label) in enumerate(dataloader):
            print('Images: \033[1;31m\033[5m%03d\033[0m/%03d' % (index + 1, len(dataloader)))
            image = image.to(args.device)
            label = label.to(args.device)

            for ratio in args.ratios:
                print('\tCurrent ratio: \033[1;31m\033[5m%.2f' % ratio)
                order = int((args.grid_size ** 2 - 2) * ratio)
                seed_torch(1000 * index + order + args.seed)
                if args.out_type == 'gt':
                    compute_order_interaction_img(args, name[0], label, ratio, interaction_logit_io_handler, interaction_io_handler)
                else:
                    raise Exception(f"output type [{args.out_type}] not supported.")


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
    parser.add_argument('--softmax_type', default='modified', type=str, choices=['normal','modified','yi'], help="reward function for interaction")
    parser.add_argument('--out_type', default='gt', type=str, choices=['gt'], help="we use the score of the ground truth class to compute interaction")
    parser.add_argument('--chosen_class', default='random', type=str, choices=['random'])
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int,
                        help="partition the input image to grid_size * grid_size patches"
                             "each patch is considered as a player")
    parser.add_argument('--no_cuda', action="store_true")


    args = parser.parse_args()

    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    seed_torch(args.seed)

    log_args_and_backup_code(args, __file__)

    interaction_logit_io_handler = InteractionLogitIoHandler(args)
    interaction_io_handler = InteractionIoHandler(args)

    model, dataloader = prepare(args,train=True)
    compute_interactions(args, model, dataloader, interaction_logit_io_handler, interaction_io_handler)
