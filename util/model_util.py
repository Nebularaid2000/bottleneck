import os
import torch
import torch.nn as nn
import torchvision

from models.alexnet import our_alexnet


def load_checkpoint(args, checkpoint_path: str, model: nn.Module, optimizer: torch.optim.Optimizer=None) -> None:
    """
    Input
        args: args
        checkpoint_path: str, path of saved model parameters
        model: nn.Module
        optimizer: torch.optim.Optimizer
    Return:
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'File doesn\'t exists {checkpoint_path}')
    print(f'=> loading checkpoint "{checkpoint_path}"')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # load models saved with legacy versions
    if not ('state_dict' in checkpoint):
        sd = checkpoint
    else:
        sd = checkpoint['state_dict']

    # load with models trained on a single gpu or multiple gpus
    if 'module.' in list(sd.keys())[0]:
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    print(f'=> loaded checkpoint "{checkpoint_path}"')
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


# ------- our models ---------

def get_our_alexnet(args, load_model=True) -> nn.Module:
    model = our_alexnet(dataset=args.dataset)
    if load_model:
        load_checkpoint(args, args.checkpoint_path, model)
    else:
        print("use model at initialization")
    return model


def get_model(args) -> nn.Module:
    """ get model and load parameters if needed
    Input:
        args: args
            if args.checkpoint_path is "None", then do not load model parameters
    Return:
        some model: nn.Module, model to be evaluated
    """
    torch.hub.set_dir(args.pretrained_models_dirname)
    if args.checkpoint_path == "None":
        load_model = False
    else:
        load_model = True

    if 'our_alexnet' in args.arch:
        print("use our alexnet")
        return get_our_alexnet(args, load_model=load_model)

    else:
        raise Exception(f"model [{args.arch}] not implemented. Error in get_model.")

