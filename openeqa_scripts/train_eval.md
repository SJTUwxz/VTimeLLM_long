# extract video features using clip model
python nextqa_scripts/feature_generation.py --data_path data/open_eqa/val-6-frames_question.json --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/data/videos/ --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/clip_features

# generate train and test annotation file
# 2-frame file annotation generation
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only True
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only False 
python openeqa_scripts/annotation.py --predict_frames_number 6 --question_only True
python openeqa_scripts/annotation.py --predict_frames_number 6 --question_only False 

# ./data/open_eqa/train-2-frames_answer.json generated
# ./data/open_eqa/val-2-frames_answer.json generated

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
exp_name=vicuna-v1-5-7b_${bs}_${lr}_${predict_frame}-frame
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

# evaluation
checkpoint_num=220
eval_predict_frame=6
dataset=openeqa
eval_subset=train
exp_dir=vicuna-v1-5-7b_8_1e-4_6-frame
dset=${dataset}_${eval_subset}
data_path=./data/open_eqa/${eval_subset}-6-frames_question.json
save_dir=ckpt-${checkpoint_num}_${eval_subset}_${eval_predict_frame}-frame-pred
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${save_dir}
mkdir -p $output_dir
mkdir -p ${output_dir}/hm3d-v0
mkdir -p ${output_dir}/scannet-v0
sbatch --gpus=1 -o ${output_dir}/%j.out -J ${dset}_${save_dir} -N 1 $SLURM_ARGS --wrap="python openeqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${dataset}_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/${dataset}/clip_features/ --save_path ${output_dir}"


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
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame
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
exp_name=${model_version}_${bs}_${lr}_${predict_frame}-frame
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

```
