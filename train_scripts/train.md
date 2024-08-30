# Stage 2 Train

```bash

ngpus=1
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2
mkdir -p ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage2 -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus}"

# multiple gpus
ngpus=8
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2-${ngpus}gpu
mkdir -p ${output_dir}
echo ${output_dir}
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage2 -N 1 $SLURM_ARGS --wrap="bash scripts/stage2.sh ${output_dir} ${ngpus}"

```

# Stage 3 train

```bash

# single gpu
ngpus=1
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage3
mkdir -p ${output_dir} 
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage3 -N 1 $SLURM_ARGS --wrap="bash scripts/stage3.sh ${output_dir} ${ngpus} ./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2"

# multiple gpus
ngpus=8
output_dir=./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage3-${ngpus}gpu
mkdir -p ${output_dir} 
sbatch --gpus=$ngpus -o ${output_dir}/%j.out -J replicate_train_stage3 -N 1 $SLURM_ARGS --wrap="bash scripts/stage3.sh ${output_dir} ${ngpus} ./checkpoints/replicated_exps/vtimellm-vicuna-v1-5-7b-stage2-${ngpus}gpu"

```
