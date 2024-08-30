#!/bin/bash

MODEL_VERSION=$1
# echo $CUDA_VISIBLE_DEVICES
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=$(( ($RANDOM % 10000) + 20000 ))
output_dir=$2
train_data=$3
batch_size=$4
learning_rate=$5
other_args=$6


deepspeed --include "localhost:${gpu_vis}" --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --training_stage 3 \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path $train_data \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --stage2_path ./checkpoints/vtimellm-$MODEL_VERSION-stage3 \
    --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ \
    --output_dir $output_dir \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate $learning_rate \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    $other_args \

    #--feat_folder /path/to/stage3_feat \
    # --save_steps 50000 \
    # --report_to none \
