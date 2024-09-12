# Stage 2 Train

```bash

ngpus=1
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage2 -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus}"

# multiple gpus
ngpus=8
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2-${ngpus}gpu
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage2 -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus}"

```

# Stage 3 train

```bash

# single gpu
ngpus=1
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage3
mkdir -p ${output_dir} 
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage3 -N 1 $SLURM_ARGS --wrap="bash scripts/stage3.sh ${output_dir} ${ngpus} ./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2"

# multiple gpus
ngpus=8
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage3-${ngpus}gpu
mkdir -p ${output_dir} 
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage3 -N 1 $SLURM_ARGS --wrap="bash scripts/stage3.sh ${output_dir} ${ngpus} ./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2-${ngpus}gpu"

```

# Stage 2 train with added temporal tokens and temporal loss

```bash

ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
loss_type=l1_loss
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}
output_dir=./checkpoints/l1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=5e-5
loss_type=l1_loss
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}
output_dir=./checkpoints/l1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""


# projecting segments to embeddings using NERF-like fourier embedding + linear
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=l1_loss
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}
output_dir=./checkpoints/l1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

# add gIOU loss
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=giou_loss
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}
output_dir=./checkpoints/l1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

# add both gIOU and L1 loss
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=l1_giou_loss
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}
output_dir=./checkpoints/l1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""
```
