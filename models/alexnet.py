import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Any


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class our_AlexNet(nn.Module): # simply put all layers in nn.Sequential

    def __init__(self, dataset):
        super(our_AlexNet, self).__init__()
        if dataset == 'tiny50':
            conv1_set = {"kernel_size":11, "stride":4, "padding":2}
            mpool_padding = 0
            fc_dims = 4096
            out_size = 6
            num_class = 50
        elif dataset == 'cifar10':
            conv1_set = {"kernel_size": 7, "stride": 1, "padding": 3}
            mpool_padding = 1
            fc_dims = 512
            out_size = 4
            num_class = 10
        else:
            raise Exception("dataset [%s] not implemented" % dataset)

        self.all_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=conv1_set["kernel_size"], stride=conv1_set["stride"], padding=conv1_set["padding"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=mpool_padding),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=mpool_padding),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=mpool_padding),

            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * out_size * out_size, fc_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.all_layers(x)
        return output

def our_alexnet(dataset):
    return our_AlexNet(dataset=dataset)