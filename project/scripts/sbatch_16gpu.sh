#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=fm-16gpu
#SBATCH -t 00:20:00
#SBATCH -N 4
#SBATCH -G 16
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -o /pscratch/sd/r/rmf253/sysml-MoE-proj/project/results/raw/fm-16gpu-%j.out
#SBATCH --export=ALL
# Four nodes, anchor only. E=32 divides 16. This is the top of the weak-scaling curve.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/bench_lib.sh"
run_case scale 16 4096 2048 2048 32 2
