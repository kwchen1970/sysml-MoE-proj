#!/bin/bash
#SBATCH --job-name=moe-build
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:20:00
#SBATCH --output=granularity-build-%j.log
set -eo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR}}"
source project/scripts/expert_granularity/env.sh
export PATH="/global/homes/k/kc994/.conda/envs/sysml/bin:$PATH"
for MODE in release trace; do
  TRACE=OFF
  if [[ "$MODE" == trace ]]; then TRACE=ON; fi
  srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 --gpu-bind=none \
    cmake -S csrc -B "csrc/cmake-build-$MODE" \
    -DCMAKE_BUILD_TYPE=Release -DFLASHMOE_PROJECT_TRACE="$TRACE"
  srun --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=4 --gpu-bind=none \
    cmake --build "csrc/cmake-build-$MODE" --target testFlashMoE --parallel 8
done
echo "Both benchmark builds completed."
