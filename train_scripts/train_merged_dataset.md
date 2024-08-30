
# Stage 2 train on momentor

## num_features_per_video=100

```bash

# 100 features per video multiple gpus
# 6 questions per conversation
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/merged_stage2_momentor.json
lr=1e-4
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --learning_rate ${lr} --num_features_per_video ${num_features_per_video}"
exp_name=merged-${num_features_per_video}-${lr}
output_dir=./checkpoints/merged_exps/${exp_name}
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""

```


## num_features_per_video=400

```bash

# 400 features per video multiple gpus
ngpus=8
num_features_per_video=400
train_data=./data/vtimellm_train/merged_stage2_momentor.json
lr=1e-4
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --learning_rate ${lr} --num_features_per_video ${num_features_per_video}"
exp_name=merged-${num_features_per_video}-${lr}
output_dir=./checkpoints/merged_exps/${exp_name}
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""


# 400 features per video multiple gpus 
#6 questions per conversation
ngpus=8
num_features_per_video=400
train_data=./data/vtimellm_train/momentor_6_questions.json
lr=1e-4
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_${num_features_per_video}/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video}"
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu-6questions
output_dir=./checkpoints/momentor_exps/${exp_name}
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""
```
