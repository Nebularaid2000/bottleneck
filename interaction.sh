inter_type="pixel";

# uncomment the setting you want to run

# normal model
gpu_id=0; dirname="result"; arch="our_alexnet_cifar10_normal_lr0.01_log1_da_flip_crop_best"; dataset="cifar10"; grid=16; seed=0;

# high-order model
#gpu_id=0; dirname="result"; arch="our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_0.5_0.0_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best"; dataset="cifar10"; grid=16; seed=0;

# mid-order model
#gpu_id=0; dirname="result"; arch="our_alexnet_cifar10_dp_pos0_deltav_baseline_0.7_0.3_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best"; dataset="cifar10"; grid=16; seed=0;

# low-order model
#gpu_id=0; dirname="result"; arch="our_alexnet_cifar10_dp_pos0_entropy_deltav_baseline_1.0_0.7_lam1.0_1.0_grid16_lr0.01_log1_da_flip_crop_best"; dataset="cifar10"; grid=16; seed=0;

python sampler.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname  --grid_size=$grid --inter_type=$inter_type --seed=$seed
python gen_pairs_pixel.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname  --grid_size=$grid --inter_type=$inter_type --seed=$seed 
python m_order_interaction_logit_pixel_baseline.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname  --grid_size=$grid --inter_type=$inter_type --seed=$seed 
python compute_interactions.py --gpu_id=$gpu_id --arch=$arch --dataset=$dataset --output_dirname=$dirname  --grid_size=$grid --inter_type=$inter_type --seed=$seed 
python draw_figures.py --arch=$arch --dataset=$dataset --output_dirname=$dirname --grid_size=$grid --inter_type=$inter_type --seed=$seed