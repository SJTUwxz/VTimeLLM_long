
# InternVid dataset

```bash

for i in `seq 1 16`; do
  sbatch -A r00371 -p gpu --nodes 1 --gpus 1 --time 02-00:00:00 --gpus 1 -J internvid/${i} --out ./${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_all_features.py \
          --video_folder ../InternVid-cut-videos-ffmpeg/ \
          --save_dir ../InternVid_clip_all_features/ \
          --data_path ../stage2.json
  "
done
```

# Charades STA dataset

```bash
python unify_scripts/charades2anet.py

# 100 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J charades/${i} --out /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/Charades_v1_480/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_100/ \
  "
done

# 400 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J charades/${i}-400 --out /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/${i}-400.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/Charades_v1_480/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_features_400/ \
          --num_features 400 \
  "
done

# 100 frames, 1 CLS + 64 patch features per frame
for i in `seq 1 4`; do
  sbatch -p h100 --gpus 1 -n 1 -J anet/clip_features/${i} --out ${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_all_features.py --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/Charades_v1_480/ --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/Charades-STA/clip_all_features_100frm/ --data_path ./data/vtimellm_eval/charades_sta_test.json
  "
done

```

# ActivityNet Captions dataset

```bash

# 100 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J anet/${i} --out /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/activity_net_2fps_360/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_100/ \
  "
done

# 400 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J anet/${i}-400 --out /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/${i}-400.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/activity_net_2fps_360/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_features_400/ \
          --num_features 400 \
  "
done

# 100 frames, 1 CLS + 64 patch features per frame
for i in `seq 1 4`; do
  sbatch -p h100 --gpus 1 -n 1 -J anet/clip_features/${i} --out ${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_all_features.py --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/activity_net_2fps_360/ --save_dir  /mnt/mir/datasets/vlm-evaluation-datasets/ActivityNet-Captions/clip_all_features_100frm --data_path ./data/vtimellm_eval/val_2.json
  "
done

```

# Momentor annotations

```bash

# all questions of one video fall into one conversation
python unify_scripts/momentor_annotations.py

# 6 questions in each conversation 
python unify_scripts/momentor_annotations_split_questions.py

python unify_scripts/momentor_annotations_split_questions.py --split_questions 2

# Extract clip features of momentor

# 100 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J momentor-10M/clip_features/${i} --out /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/videos/ \
          --save_dir /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features/ \
  "
done

# 400 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J momentor-10M/clip_features/${i} --out /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_400/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-datasets/momentor-10M/videos/ \
          --save_dir /mnt/mir/datasets/vlm-datasets/momentor-10M/clip_features_400/ \
          --num_features 400 \
  "
done


```

# QVHighlights dataset

```bash

# 100 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J qvhighlights/${i} --out /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/${i}.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/videos/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/clip_features_100/ \
  "
done

# 400 features
for i in `seq 1 24`; do
  sbatch --gpus 1 -n 1 -J qvhighlights/${i}-400 --out /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/${i}-400.out \
      $SLURM_ARGS --wrap="python unify_scripts/clip_features.py \
          --video_folder /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/videos/ \
          --save_dir /mnt/mir/datasets/vlm-evaluation-datasets/QVHighlights/clip_features_400/ \
          --num_features 400 \
  "
done

```
