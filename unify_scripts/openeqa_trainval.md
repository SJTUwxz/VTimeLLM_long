# extract video features using clip model
python nextqa_scripts/feature_generation.py --data_path data/open_eqa/val-6-frames_question.json --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/data/videos/ --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/clip_features

# generate train and test annotation file
# 2-frame file annotation generation
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only True
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only False 
python openeqa_scripts/annotation.py --predict_frames_number 6 --question_only True
python openeqa_scripts/annotation.py --predict_frames_number 6 --question_only False 

## 1.a save multiple best segments generated from best frames
python openeqa_scripts/annotation_segment.py --question_only False 
python openeqa_scripts/annotation_segment.py --question_only True

## 1.b save multiple pseudo best segments to pickles

python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file train-segment_question.json --save_dir open_eqa_segments
python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file val-segment_question.json --save_dir open_eqa_segments

## 1.c multiple segments saved folder:
/mnt/opr/fengchan/dataset/sparse_bwd/psuedo_selected_frames/open_eqa_segments/ 

## 2.a save one best segment to annotation json file
## best segment is the one where the best first frame is in
## train_bestsegment_[question, answer].json
python openeqa_scripts/annotation_most_important_segment.py --question_only False 
python openeqa_scripts/annotation_most_important_segment.py --question_only True

## 2.b save the most important segment to pickles
python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file train-bestsegment_question.json --save_dir open_eqa_bestsegments
python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file val-bestsegment_question.json --save_dir open_eqa_bestsegments

## 2.c save dir
/mnt/opr/fengchan/dataset/sparse_bwd/psuedo_selected_frames/open_eqa_bestsegments/ 


# OpenEQA 

## script for 6-frames prediction

```bash

# training lr=1e-4 question only
predict_frame=6
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
train_data=./data/open_eqa/train-${predict_frame}-frames_question.json
bs=8
lr=1e-4
exp_name=vicuna-v1-5-7b_${bs}_${lr}_${predict_frame}-frame_question
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 200 --save_steps 20" 
mkdir -p $output_dir
sbatch --gpus=8 -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh vicuna-v1-5-7b $output_dir $train_data $bs $lr \"$other_args\""

# training lr=1e-4 question and answer
predict_frame=6
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
train_data=./data/open_eqa/train-${predict_frame}-frames_answer.json
bs=8
lr=1e-4
exp_name=vicuna-v1-5-7b_${bs}_${lr}_${predict_frame}-frame_answer
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=8 -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh vicuna-v1-5-7b $output_dir $train_data $bs $lr \"$other_args\""

# generate frames for train and val dataset

for q_or_a in question answer; do
eval_subset=train
checkpoint_num=220
lr=1e-4
predict_frame=6
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_6-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-6-frames_${q_or_a}.json
save_dir=${eval_subset}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

for q_or_a in question answer; do
eval_subset=val
checkpoint_num=220
lr=1e-4
predict_frame=6
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_6-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-6-frames_${q_or_a}.json
save_dir=${eval_subset}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

### try lowering the learning rate to 1e-5 with question only model
predict_frame=6
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
train_data=./data/open_eqa/train-${predict_frame}-frames_question.json
bs=8
lr=1e-5
exp_name=vicuna-v1-5-7b_${bs}_${lr}_${predict_frame}-frame_question
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 200 --save_steps 20" 
mkdir -p $output_dir
sbatch --gpus=8 -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh vicuna-v1-5-7b $output_dir $train_data $bs $lr \"$other_args\""


```

## script for 2-frames prediction

```bash

# training lr=1e-4 question only
predict_frame=2
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-${predict_frame}-frames_question.json
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame_question
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# training lr=2e-5 question only
predict_frame=2
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-${predict_frame}-frames_question.json
bs=8
lr=2e-5
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame_question
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# training lr=1e-4 question and answer
predict_frame=2
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-${predict_frame}-frames_answer.json
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame_answer
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# training lr=2e-5 question and answer
predict_frame=2
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-${predict_frame}-frames_answer.json
bs=8
lr=2e-5
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame_answer
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# evaluation lr=1e-4

for q_or_a in question answer; do
eval_subset=train
checkpoint_num=400
lr=1e-4
predict_frame=2
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_${predict_frame}-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-${predict_frame}-frames_${q_or_a}.json
save_dir=${eval_subset}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

for q_or_a in question answer; do
eval_subset=val
checkpoint_num=400
lr=1e-4
predict_frame=2
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_${predict_frame}-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-${predict_frame}-frames_${q_or_a}.json
save_dir=${eval_subset}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

# evaluation lr=2e-5

for q_or_a in question answer; do
eval_subset=train
checkpoint_num=500
lr=2e-5
predict_frame=2
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_${predict_frame}-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-${predict_frame}-frames_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

for q_or_a in question answer; do
eval_subset=val
checkpoint_num=500
lr=2e-5
predict_frame=2
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_${predict_frame}-frame_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-${predict_frame}-frames_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_${predict_frame}-frame_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"
done

```

## script for best segments prediction

```bash

# TRAINING lr=1e-4 question answer 
# for q_or_a in question answer; do
for q_or_a in answer; do
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-bestsegment_${q_or_a}.json
bs=8
lr=5e-5
exp_name=${model_version}_${bs}_${lr}_bestsegment_${q_or_a}
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
done

# EVALUATION choose checkpoint

for checkpoint_num in `seq 100 10 300`; do
q_or_a=answer
eval_subset=train
lr=5e-5
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_bestsegment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-bestsegment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_bestsegment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

for checkpoint_num in `seq 100 10 300`; do
q_or_a=answer
eval_subset=val
lr=5e-5
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_bestsegment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-bestsegment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_bestsegment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

```

## script for multiple segments prediction

```bash

# training lr=1e-4 question answer 
# for q_or_a in question answer; do
for q_or_a in answer; do
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-segment_${q_or_a}.json
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}_segment_${q_or_a}
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 10" 
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
done

# predict and save 6 frames on train and val set

for q_or_a in question answer; do
eval_subset=train
checkpoint_num=200
lr=1e-4
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_segment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-segment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_segment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

for q_or_a in question answer; do
eval_subset=val
checkpoint_num=200
lr=1e-4
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_segment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-segment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_segment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

### try lower the learning rate to 5e-5 question only
# for q_or_a in question answer; do
for q_or_a in answer; do
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version=vicuna-v1-5-7b
train_data=./data/open_eqa/train-segment_${q_or_a}.json
bs=8
lr=5e-5
exp_name=${model_version}_${bs}_${lr}_segment_${q_or_a}
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 100 --save_steps 20" 
rm -rf $output_dir
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
done

# lr: 5e-5 predict and save 6 frames on train and val set

for q_or_a in question answer; do
eval_subset=train
checkpoint_num=240
lr=5e-5
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_segment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-segment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_segment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

for q_or_a in question answer; do
eval_subset=val
checkpoint_num=240
lr=5e-5
dataset=openeqa
exp_dir=vicuna-v1-5-7b_8_${lr}_segment_${q_or_a}
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-segment_${q_or_a}.json
save_dir=${eval_subset}_${lr}_ckpt-${checkpoint_num}_segment_${q_or_a}
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python unify_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir} --predict_segment"
done

```

#### calculate 2/6-frame prediction normalized distance

```bash

python unify_scripts/calculate_frame_distance_openeqa.py --json_file train-2-frames_answer.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file train-2-frames_question.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file train-6-frames_answer.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file train-6-frames_question.json

python unify_scripts/calculate_frame_distance_openeqa.py --json_file val-2-frames_answer.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file val-2-frames_question.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file val-6-frames_answer.json
python unify_scripts/calculate_frame_distance_openeqa.py --json_file val-6-frames_question.json

```
#### from psuedo best segments save to pseudo selected frames

```bash

## multiple pseudo best segments

python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file train-segment_question.json
python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file val-segment_question.json


## most important / best segment where the best first frame is in

python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file train-bestsegment_question.json

python unify_scripts/openeqa_groundtruth_segment2frame.py --json_file val-bestsegment_question.json


```

