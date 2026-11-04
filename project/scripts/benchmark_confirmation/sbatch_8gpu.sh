#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=fm-8gpu
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH -G 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -o /pscratch/sd/r/rmf253/sysml-MoE-proj/project/results/raw/fm-8gpu-%j.out
#SBATCH --export=ALL
# Two Perlmutter nodes (Slingshot 11 between them). Same token and expert
# axes as the 4-GPU job so the paper's 4-GPU and 8-GPU panels can be drawn.
# E in {8,16,32,128} all divide 8. The E=32 anchor is the S=4096 token point.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/bench_lib.sh"

for S in 1024 4096 8192 16384; do
  run_case tokens 8 "$S" 2048 2048 32 2
done
for E in 8 16 128; do
  run_case experts 8 4096 2048 2048 "$E" 2
done
