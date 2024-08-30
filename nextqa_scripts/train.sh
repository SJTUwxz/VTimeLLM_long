
# vicuna-7b  lr=1e-4 run for 10 epochs
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/nextqa_train.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-debug
output_dir=$res_dir/${dset}/${exp_name}
other_args=""
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""


# vicuna-7b  lr=5e-5 run for 10 epochs
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/nextqa_train.json'
bs=8
lr=5e-5
exp_name=${model_version}_${bs}_${lr}-debug
output_dir=$res_dir/${dset}/${exp_name}
other_args=""
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# vicuna-7b  lr=1e-5 run for 10 epochs
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/nextqa_train.json'
bs=8
lr=1e-5
exp_name=${model_version}_${bs}_${lr}-debug
output_dir=$res_dir/${dset}/${exp_name}
other_args=""
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# start from stage2 pretrained vicuna-7b  lr=1e-4 run for 10 epochs
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/nextqa_train.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-stage2
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# set as grounding task: let input have both question and answer, and add more instructions to the instruction prompt
# predicting frames only
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_with_answer.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-stage2-answer
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# question only, and add more instructions to the instruction prompt
# predicting both answers and frames
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-stage2-question
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# question only, predicting 6 frames only and don't predict the answer
ngpus=1
node=node001
visible_gpu=0
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_6-frame_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-stage2-question-predictframes
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -w ${node} -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="CUDA_VISIBLE_DEVICES=${visible_gpu} bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""


# question only, predict 1 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_1-frame_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-question-predict1
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""


# question and answer, predict 1 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_1-frame_answer.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-answer-predict1
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# question only, predict 2 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_2-frame_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-question-predict2
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# question and answer, predict 2 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_2-frame_answer.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-answer-predict2
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# question only, predict 3 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_3-frame_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-question-predict3
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
# sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr "$other_args"

# question and answer, predict 3 frame
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_3-frame_answer.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-answer-predict3
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
# sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr "$other_args"

# question only predict segments
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_segment_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-question-segment
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2" 
mkdir -p $output_dir
# sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr "$other_args"

# question and answer predict segments
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_segment_answer.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-answer-segment
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2 --num_train_epochs 200" 
mkdir -p $output_dir
# sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr "$other_args"

# overfitting experiment: not giving video frames, feed in the video_id, question_id
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_id-only-20samples.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-id-only-20samples
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2 --num_train_epochs 200 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
# bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr "$other_args"

ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_id-only-20samples.json'
bs=8
lr=1e-3
exp_name=${model_version}_${bs}_${lr}-id-only-20samples
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2 --num_train_epochs 200 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""

# overfitting experiment with video tokens put back
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models/
dset=next_qa_train
model_version="vicuna-v1-5-7b"
train_data='./data/next_qa_train_6-frame_question_20.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}-20samples
output_dir=$res_dir/${dset}/${exp_name}
other_args="--stage2_path ./checkpoints/vtimellm-${model_version}-stage2 --num_train_epochs 200 --save_steps 10" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./nextqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
