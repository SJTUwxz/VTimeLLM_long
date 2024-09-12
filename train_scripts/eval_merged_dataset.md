
# Evaluation on ActivityNet

```bash

python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_eval/val_2.json \
     --log_path ${output_dir}/output.log \

for checkpoint in `seq 5000 500 14500`; do
ngpus=8
num_features_per_video=100
lr=1e-4
exp_name=merged-${num_features_per_video}-${lr}
output_dir=./checkpoints/merged_results/${exp_name}-ckpt${checkpoint}-eval
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/merged_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done


```

# Charades-STA evaluation

```bash
python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_eval/charades_sta_test.json \
     --log_path ${output_dir}/output.log \

for checkpoint in `seq 5000 500 14000`; do
ngpus=8
num_features_per_video=100
lr=1e-4
exp_name=merged-${num_features_per_video}-${lr}
output_dir=./checkpoints/merged_results/${exp_name}-ckpt${checkpoint}-eval_Charades
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/charades_sta_test.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/merged_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done

```
