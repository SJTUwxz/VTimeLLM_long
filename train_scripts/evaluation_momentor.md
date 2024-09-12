# Evaluation using trained checkpoint on ActivityNet

```bash

python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_eval/val_2.json \
     --log_path ${output_dir}/output.log \

# 100 features 
# 1 gpus
# does not split questions into different conversations
for checkpoint in `seq 500 500 1500`; do
ngpus=8
num_features_per_video=100
lr=1e-4
exp_name=momentor-${num_features_per_video}-${lr}
output_dir=./checkpoints/momentor_results/${exp_name}-ckpt${checkpoint}-eval
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/momentor_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done



# 400 features 
# 8 gpus
# 6 questions per conversation
for checkpoint in `seq 2000 500 4500`; do
ngpus=8
num_features_per_video=400
lr=1e-4
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu-6questions
output_dir=./checkpoints/momentor_results/${exp_name}-ckpt${checkpoint}-eval
rm -rf ${output_dir}
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/momentor_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done


# 100 features 
# 8 gpus
# 6 questions per conversation
for checkpoint in `seq 2000 500 6000`; do
ngpus=8
num_features_per_video=100
lr=1e-4
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu-6questions
output_dir=./checkpoints/momentor_results/${exp_name}-ckpt${checkpoint}-eval
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/momentor_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done


# 400 features 
# 8 gpus
# does not limit the number of questions per conversation
for checkpoint in `seq 500 500 2500`; do
ngpus=8
num_features_per_video=400
lr=1e-4
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu
output_dir=./checkpoints/momentor_results/${exp_name}-ckpt${checkpoint}-eval
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/val_2.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/momentor_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done

```

# Charades-STA evaluation
```bash
python vtimellm/eval/metric.py \
     --data_path ./data/vtimellm_eval/charades_sta_test.json \
     --log_path ${output_dir}/output.log \

for checkpoint in `seq 2000 500 6000`; do
ngpus=8
num_features_per_video=100
lr=1e-4
exp_name=momentor-${num_features_per_video}-${lr}-${ngpus}gpu-6questions
output_dir=./checkpoints/momentor_results/${exp_name}-ckpt${checkpoint}-eval_Charades
mkdir -p $output_dir
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${exp_name}-ckpt${checkpoint}-eval -N 1 $SLURM_ARGS --wrap="python vtimellm/eval/eval.py --data_path ./data/vtimellm_eval/charades_sta_test.json --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_${num_features_per_video} --log_path ${output_dir}/output.log --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 checkpoints/momentor_exps/${exp_name}/checkpoint-${checkpoint}/ --num_features_per_video ${num_features_per_video}"
done

```
