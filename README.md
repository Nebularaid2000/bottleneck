# Representation Bottleneck

## Overview

This repository is an implementation of the paper **Discovering and Explaining the Representation Bottleneck of DNNs** ([arxiv](https://arxiv.org/abs/2111.06236)), which was accepted as an Oral presentation at ICLR 2022.



## Requirements

- Python 3.9
- pytorch 1.7.1
- CUDA 11.0
- numpy 1.19.5
- torchvision 0.8.2

All models were trained on a single NVIDIA GeForce RTX 3090 GPU.



## Usage

### Training

Run the following shell script to train the models:

```shell
./train.sh
```

You can change the gpu by changing the `--gpu_id` argument in the script.

The models are saved in the `checkpoints` directory by default.

### Compute interaction

Run the following shell script to compute interaction for the models:

```
./interaction.sh
```

You can uncomment the setting you want to run on top of the script.

The results are saved in the `results` directory by default.



## Citation

If you use this project in your research, please cite it.

```
@inproceedings{
  deng2022discovering,
  title={{Discovering} and {Explaining} the {Representation} {Bottleneck} of {DNNs}},
  author={Huiqi Deng and Qihan Ren and Hao Zhang and Quanshi Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

