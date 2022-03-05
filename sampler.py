import os
import random
import torch
import numpy as np
import torchvision
import argparse

from common import config
from io_handler import SampleIoHandler, set_args
from util.utils import seed_torch, prepare, normalize, log_args_and_backup_code


def sample(args):
    """ random sample some images from dataset
    if chosen_class is "random":
        - cifar10: select the first args.selected_img_number//num_classes images in each class (order is determined by the dataloader)
    """
    if args.dataset == "cifar10":
        cifar10_train = torchvision.datasets.CIFAR10(root=os.path.join(args.prefix, args.datasets_dirname), train=True, download=True)
        all_labels = np.array(cifar10_train.targets) # labels of all imgs, in order
        selected_indices = []
        selected_class_names = []
        selected_labels = []
        if args.selected_img_number < args.class_number:
            # if the number of images is less than the number of class
            # then we just choose the first image of the first selected_img_number classes
            selected_class = args.selected_img_number
            img_num_per_class = np.array([1] * selected_class)
        else:
            selected_class = args.class_number
            img_num_per_class = np.array([args.selected_img_number // selected_class] * selected_class)
            remainder = args.selected_img_number % selected_class
            for r in range(remainder):  # if there is remainder, then each class whose id is less than remainder will have an extra image
                img_num_per_class[r] += 1

        for c in range(selected_class):
            print("class %d" % c)
            indices = np.nonzero(all_labels == c)[0]
            selected_indices.extend(indices[:img_num_per_class[c]].tolist()) # choose the first img_num_per_class images
            selected_class_names.extend([cifar10_train.classes[c]] * img_num_per_class[c])
            selected_labels.extend([c] * img_num_per_class[c])
        assert args.selected_img_number == len(selected_indices)
        print("Select random images done.")
        return [(selected_class_names[index], selected_indices[index], selected_labels[index])
                for index in range(args.selected_img_number)]
    else:
        raise Exception(f"Dataset [{args.dataset}] not implemented. Error in sampler.")


def check_if_correct_cls(args, model, dataloader, sample_list): # check whether classify correctly
    model.to(args.device)
    count = 0  # num of correct classifications
    sample_list_selected = []
    with torch.no_grad():
        model.eval()
        for index, (name, data, label) in enumerate(dataloader):
            data = data.to(args.device)
            label = label.to(args.device)
            if "celeba" in args.dataset:
                label = label[:, args.celeba_clf_attr]
            print("img: %d " % index, name, 'label:', label)
            data = normalize(args, data) # this has no effect on tabular data

            output = model(data)
            pred = torch.argmax(output, dim=1)
            print('pred:', pred.item())

            if pred.item() == label.item():
                count += 1
                sample_list_selected.append(sample_list[index])
            else:
                print('Predict incorrectly.')
            print('----------------------------')

    print(count, 'images/tabular data are classified correctly.')
    return sample_list_selected


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
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID")
    parser.add_argument('--chosen_class', default='random', type=str, choices=['random'])
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('--grid_size', default=16, type=int)

    args = parser.parse_args()
    set_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_torch(args.seed)

    if not os.path.exists(args.samples_dir):
        os.makedirs(args.samples_dir)

    log_args_and_backup_code(args, __file__)

    # random sample images
    sample_all_save_path = os.path.join(args.samples_dir, "samples_all.txt")
    sample_list = sample(args)
    with open(sample_all_save_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(map(lambda item: f'{item[0]},{item[1]},{item[2]}', sample_list)))

    # only keep the images that can be correctly classified
    args.samples_file = sample_all_save_path # This step is important. When load datasets, we will use samples_all.txt instead of samples_selected.txt
    model, dataloader = prepare(args, train=True) # load all random sampled images
    print("prepared")
    sample_list_selected = check_if_correct_cls(args, model, dataloader, sample_list) # images that are correctly classified

    args.samples_file = os.path.join(args.samples_dir, config['samples_filename']) # change the file name back, for saving imgs that are correctly classified
    sample_io_handler = SampleIoHandler(args)
    sample_io_handler.save(sample_list_selected) # only save images correctly classified to samples_selected.txt
