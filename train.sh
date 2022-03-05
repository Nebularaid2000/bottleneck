# normal model
python trainer.py --arch=our_alexnet --dataset=cifar10 --gpu_id=0
# high-order model
python trainer_dropout_entropy_deltav_baseline.py --arch=our_alexnet --dataset=cifar10 --preserve_rate1=0.5 --preserve_rate2=0.0 --gpu_id=0
# mid-order model
python trainer_dropout_deltav_baseline.py --arch=our_alexnet --dataset=cifar10 --preserve_rate1=0.7 --preserve_rate2=0.3 --gpu_id=0
# low-order model
python trainer_dropout_entropy_deltav_baseline.py --arch=our_alexnet --dataset=cifar10 --preserve_rate1=1.0 --preserve_rate2=0.7 --gpu_id=0