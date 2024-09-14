# Extract clip features of momentor

```bash
# 100 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J momentor-10M/clip_features/${i} --out /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/videos/ \
          --save_dir /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features/ \
  "
done

# 400 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J momentor-10M/clip_features/${i} --out /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_400/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/videos/ \
          --save_dir /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_400/ \
          --num_features 400 \
  "
done

# 100 frames, 1 CLS + 64 patch features per frame
for i in `seq 1 4`; do
  sbatch -p h100 --gpus 1 -n 1 -J momentor-10M/clip_features/${i} --out {i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/check_momentor_clip_feat.py --video_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/videos/ --save_dir /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_all_features_100frm/ 
  "
done

```

# Stage 2 train on momentor

## num_features_per_video=100

```bash


# 100 features per video multiple gpus
# 6 questions per conversation
ngpus=8
num_features_per_video=100
train_data=./data/vtimellm_train/momentor_6_questions.json
lr=1e-4
other_args="--num_train_epochs 5 --save_steps 500 --data_path ${train_data} --feat_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_${num_features_per_video}/ --learning_rate ${lr} --num_features_per_video ${num_features_per_video}"
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu-6questions
output_dir=./checkpoints/momentor_exps/${exp_name}
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus} \"$other_args\""


```


