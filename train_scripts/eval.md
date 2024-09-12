# Evaluation using released checkpoint

```bash

# stage 3
ngpus=8
exp_name=stage3_released_checkpoint
output_dir=./checkpoints/replicated_exps/${exp_name}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder ./data/vtimellm_train/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage3 checkpoints/vtimellm-vicuna-v1-5-7b-stage3"

python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_train/val_2.json \
     --log_path ./log/released_checkpoint_output.log \

# stage 2
ngpus=8
exp_name=stage2_released_checkpoint
output_dir=./checkpoints/replicated_exps/${exp_name}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder ./data/vtimellm_train/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/"

python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_train/val_2.json \
     --log_path ./log/released_checkpoint_output.log \

```

# Evaluation using trained checkpoint

```bash

# stage 2 8 gpu
ngpus=8
exp_name=stage2_trained_ckpt_8gpu_eval
output_dir=./checkpoints/replicated_exps/${exp_name}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder ./data/vtimellm_train/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2-8gpu"

python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_train/val_2.json \
     --log_path ./log/released_checkpoint_output.log \

# stage 2 single gpu
ngpus=8
exp_name=stage2_trained_ckpt_eval
output_dir=./checkpoints/replicated_exps/${exp_name}
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder ./data/vtimellm_train/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2"



```

# Charades-STA evaluation
```bash
python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_eval/charades_sta_test.json \
     --log_path ${output_dir}/output.log \


ngpus=8
exp_name=stage3_released_checkpoint
output_dir=./checkpoints/replicated_exps/${exp_name}-charades
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/charades_sta_test.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_100/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage3 checkpoints/vtimellm-vicuna-v1-5-7b-stage3"

ngpus=8
exp_name=stage2_released_checkpoint
output_dir=./checkpoints/replicated_exps/${exp_name}-charades
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/charades_sta_test.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_100/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/"

ngpus=1
exp_name=stage2_trained_ckpt_eval
output_dir=./checkpoints/replicated_exps/${exp_name}-charades
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/charades_sta_test.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_100/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2"

```

# Evaluate the added temporal losses because we need to extract the temporal output embeddings

```bash
ngpus=1
exp_name=stage2_trained_ckpt_eval
output_dir=./checkpoints/replicated_exps/${exp_name}-debug
temporal_loss=False
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder ./data/vtimellm_train/stage3_clip_feat/ --log_path ${output_dir}/released_checkpoint_output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2 --temporal_loss ${temporal_loss}"
```
