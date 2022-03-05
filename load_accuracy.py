import torch
import numpy as np
import matplotlib.pyplot as plt
import os

normal_alexnet = torch.load("checkpoints/our_alexnet_cifar10/normal_lr0.01_log1_da_flip_crop_seed0/training_stats.pth", map_location="cpu")
alexnet_high_order = torch.load("checkpoints/our_alexnet_cifar10/dp_pos0_entropy_deltav_baseline_0.5_0.0_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/training_stats.pth", map_location="cpu")
alexnet_mid_order = torch.load("checkpoints/our_alexnet_cifar10/dp_pos0_deltav_baseline_0.7_0.3_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/training_stats.pth", map_location="cpu")
alexnet_low_order = torch.load("checkpoints/our_alexnet_cifar10/dp_pos0_entropy_deltav_baseline_1.0_0.7_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_seed0/training_stats.pth", map_location="cpu")

print("Alexnet cifar10 accuracy")
print("normal model, best acc: ", max(normal_alexnet["val_acc"]))
print("low-order model, best acc: ", max(alexnet_low_order["val_ori_acc"]))
print("mid-order model, best acc: ", max(alexnet_mid_order["val_ori_acc"]))
print("high-order model, best acc: ", max(alexnet_high_order["val_ori_acc"]))

