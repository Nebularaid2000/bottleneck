import os
from typing import List, Tuple

import torchvision
from torch.utils.data import Dataset, Subset
from io_handler import SampleIoHandler
from PIL import Image
import numpy as np

class CIFAR10_selected(Dataset):

    def __init__(self, args, root: str, transform: torchvision.transforms.Compose, train: bool) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.selected_imgs = SampleIoHandler(args).load()  # list of (class_name, img index(in the WHOLE dataset, not in a specific class, 0-based), class_index)
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        self.class_name_list = self.dataset.classes # list of class names

    def __getitem__(self, index):
        img_index_in_whole_dataset = self.selected_imgs[index][1]
        image, label = self.dataset[img_index_in_whole_dataset] # call the __getitem__ method of dataset, label is int
        assert label == self.selected_imgs[index][2]
        assert self.class_name_list[label] == self.selected_imgs[index][0]
        name = self.class_name_list[label] + "_%05d" % img_index_in_whole_dataset # e.g. airplane_00029
        return name, image, label

    def __len__(self):
        return len(self.selected_imgs)


def get_dataset_util(args, transform: torchvision.transforms.Compose, train: bool):
    """ get dataset
    Input:
        args:
        transform: torchvision.transforms.Compose, transform for the image; tabular data do not need transform
        train: bool, only valid when dataset is NOT ImageNet.
            If train=False, use the validation set. If train=True, use the training set.
            By default we will use the training set. When evaluating on ImageNet, we have to use the validation set.
    Return:
        some dataset: Dataset,
    """
    if args.dataset == "cifar10":
        root = os.path.join(args.prefix, args.datasets_dirname)
        return CIFAR10_selected(args, root, transform, train)
    else:
        raise Exception(f"dataset [{args.dataset}] not implemented. Error in get_dataset_util")



