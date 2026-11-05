#!/bin/bash
#SBATCH --job-name=gpu4-check
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:05:00
#SBATCH --output=gpu4-check-%j.log

set -eo pipefail

srun --ntasks=4 --gpus-per-task=1 --gpu-bind=closest \
  bash -c 'echo "rank=$SLURM_PROCID host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=index,name --format=csv,noheader'
