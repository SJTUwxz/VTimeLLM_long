# val dataset
# vicuna-v1-5-7b_8_1e-4-stage2
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2
dset=next_qa
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-answer
dset=next_qa_6-frame_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


# predict frames and answer
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-question
dset=next_qa_6-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

# predict frames only
exp_dir=vicuna-v1-5-7b_8_1e-4-stage2-question-predictframes
dset=next_qa_6-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

# ablate on the number of frames to predict
exp_dir=vicuna-v1-5-7b_8_1e-4-question-predict1
dset=next_qa_1-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-answer-predict1
dset=next_qa_1-frame_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-question-predict2
dset=next_qa_2-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-answer-predict2
dset=next_qa_2-frame_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-question-predict3
dset=next_qa_3-frame_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-answer-predict3
dset=next_qa_3-frame_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

### predict segment ##########3
exp_dir=vicuna-v1-5-7b_8_1e-4-question-segment
dset=next_qa_segment_question
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

exp_dir=vicuna-v1-5-7b_8_1e-4-answer-segment
dset=next_qa_segment_answer
data_path=./data/${dset}.json
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"


# over fitting experiment on video id and question id only
exp_dir=vicuna-v1-5-7b_8_1e-3-id-only
dset=next_qa_train_id-only
data_path=./data/${dset}.json
checkpoint=checkpoint-530
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir}/${checkpoint} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"

# overfitting experiment on video frames as input
exp_dir=vicuna-v1-5-7b_8_1e-3-20samples
dset=next_qa_train_6-frame_question_20
data_path=./data/${dset}.json
checkpoint=checkpoint-140
stage_num=2
output_dir=/home/fengchan/stor/dataset/sparse_bwd/vtimellm_predictions/${dset}/${exp_dir}
mkdir -p $output_dir
ngpus=1
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J ${exp_dir} -N 1 $SLURM_ARGS --wrap="python nextqa_scripts/eval_save_prediction.py --stage3 /home/fengchan/stor_arc/workspace/long-video-vlm/xizi/vtimellm/frame_selection_models/${exp_dir}/${checkpoint} --data_path ${data_path} --feat_folder /mnt/mir/datasets/vlm-evaluation-datasets/nextqa/clip_features/ --task nextqa --model_base ./checkpoints/vicuna-7b-v1.5/ --stage2 ./checkpoints/vtimellm-vicuna-v1-5-7b-stage${stage_num}/ --save_path ${output_dir}"
