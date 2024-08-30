# extract video features using clip model
python nextqa_scripts/feature_generation.py --data_path data/open_eqa/val-6-frames_question.json --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/data/videos/ --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/clip_features

# generate train and test annotation file
# 2-frame file annotation generation
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only True
python openeqa_scripts/annotation.py --predict_frames_number 2 --question_only False 

# ./data/open_eqa/train-2-frames_answer.json generated

# start from stage2 pretrained vicuna-7b  lr=1e-4 run for 10 epochs
#
ngpus=8
res_dir=$LVLM_EXP_DIR/xizi/vtimellm/frame_selection_models
dset=openeqa_train
model_version="vicuna-v1-5-7b"
train_data='./data/open_eqa/train-6-frames_question.json'
bs=8
lr=1e-4
exp_name=${model_version}_${bs}_${lr}
output_dir=$res_dir/${dset}/${exp_name}
other_args="--num_train_epochs 200 --save_steps 20" 
mkdir -p $output_dir
sbatch --gpus=$ngpus -o $output_dir/%j.out -J ${exp_name} -N 1 $SLURM_ARGS --wrap="bash ./openeqa_scripts/stage3.sh $model_version $output_dir $train_data $bs $lr \"$other_args\""
