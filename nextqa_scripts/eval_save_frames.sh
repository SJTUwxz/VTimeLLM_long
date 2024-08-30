# vicuna-v1-5-7b_8_1e-4-stage2
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2
dset=next_qa
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vila/frame_selection_exps/next_qa_train/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-answer
dset=next_qa_6-frame_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vila/frame_selection_exps/next_qa_train/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


# predict frames and answer
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-question
dset=next_qa_6-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vila/frame_selection_exps/next_qa_train/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

# predict frames only
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-question-predictframes
dset=next_qa_6-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vila/frame_selection_exps/next_qa_train/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


# test on subset train set

exp_dir=vicuna-v1-5-7b_8_1e-4-stage2
dset=next_qa_traintest
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/predicted_best_frames/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vila/frame_selection_exps/next_qa_train/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"
