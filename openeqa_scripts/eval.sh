ngpus=1
exp_dir=vicuna-v1-5-7b_8_1e-4
dset=openeqa_val
data_path=./data/open_eqa/train-6-frames_question.json
stage_num=2
checkpoint_num=220
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python openeqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/openeqa_train/${exp_dir}/checkpoint-${checkpoint_num} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/openeqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

