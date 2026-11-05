#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --job-name=fm-4gpu
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH -o /pscratch/sd/r/rmf253/sysml-MoE-proj/project/results/raw/fm-4gpu-%j.out
#SBATCH --export=ALL
# Single-node paper sweeps. Four points on each axis; the anchor is not repeated.
# Fixed within a sweep: H=2048, k=2, and whichever of S, E, I is not the axis.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/bench_lib.sh"

for S in 1024 4096 8192 16384; do
  run_case tokens 4 "$S" 2048 2048 32 2
done
for E in 8 16 128; do
  run_case experts 4 4096 2048 2048 "$E" 2
done
for I in 1024 4096 8192; do
  run_case ffn 4 4096 2048 "$I" 32 2
done
