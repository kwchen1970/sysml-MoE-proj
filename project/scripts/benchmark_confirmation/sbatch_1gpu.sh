#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=fm-1gpu
#SBATCH -t 00:20:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o /pscratch/sd/r/rmf253/sysml-MoE-proj/project/results/raw/fm-1gpu-%j.out
# One GPU, paper anchor. Other axes stay fixed so this point sits on the scaling curve.
#SBATCH --export=ALL

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/bench_lib.sh"
# H=2048 I=2048 E=32 k=2 S=4096. E=32 divides 1, 4, 8, and 16.
run_case scale 1 4096 2048 2048 32 2
# 800 eager iterations so a 1-second nvidia-smi sample can see the kernel.
# The 4-iteration sweeps finish in a few milliseconds and undersample SM util.
run_case util 1 4096 2048 2048 32 2 0 8 800
