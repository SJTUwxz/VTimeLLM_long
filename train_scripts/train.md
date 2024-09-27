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

ngpus=8
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
other_args="--num_train_epochs 6 --save_steps 1000 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr}"
exp_name=vtimellm-vicuna-v1-5-7b-stage2-${ngpus}gpu
output_dir=./checkpoints/replicated_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

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

# simple linear l1_loss predict start and end
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
loss_type=l1_loss
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${loss_type}
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

# simple linear gIoU loss predict start and end
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
loss_type=giou_loss
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${loss_type}
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

# simple linear l1_loss predict center and offset
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
loss_type=l1_loss
predict_center_offset=True
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type} --predict_center_offset ${predict_center_offset}"
exp_name=vtimellm-stage2-${loss_type}-centeroffset
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

# simple linear gIoU loss predict center and offset
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
loss_type=giou_loss
predict_center_offset=True
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --loss_type ${loss_type} --predict_center_offset ${predict_center_offset}"
exp_name=vtimellm-stage2-${loss_type}-centeroffset
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

#### sin-cos embedding

# projecting segments to embeddings using NERF-like fourier embedding + linear
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=l1_loss
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${projector_type}-${loss_type}
output_dir=./checkpoints/v1_loss_exps/${exp_name}
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
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}
output_dir=./checkpoints/v1_loss_exps/${exp_name}
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
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""



# predict center and offset, not start and end, l1_loss

ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=l1_loss
predict_center_offset=True
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type} --predict_center_offset ${predict_center_offset}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}-centeroffset
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""


# predict center and offset, gIoU loss
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/stage2.json
lr=1e-4
projector_type=angular
loss_type=giou_loss
predict_center_offset=True
other_args="--num_train_epochs 3 --save_steps 500 --data_path ${train_data} --feat_folder ./data/vtimellm_train/intern_clip_feat/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video} --add_temporal_tokens True --temporal_loss True --projector_type ${projector_type} --loss_type ${loss_type} --predict_center_offset ${predict_center_offset}"
exp_name=vtimellm-stage2-${num_features_per_video}-${lr}-${projector_type}-${loss_type}-centeroffset
output_dir=./checkpoints/v1_loss_exps/${exp_name}
rm -rf $output_dir
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

```
