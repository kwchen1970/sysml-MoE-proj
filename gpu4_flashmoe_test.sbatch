#!/bin/bash
#SBATCH --job-name=flashmoe-4gpu-test
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH --output=flashmoe-4gpu-test-%j.log
set -eo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR}}"
source project/scripts/expert_granularity/env.sh
export PATH="/global/homes/k/kc994/.conda/envs/sysml/bin:$PATH"
# Build the changed host code inside this allocation, never on login-node GPUs.
# Default rebuild prevents accidentally testing the old device=0 executable.
if [[ "${REBUILD:-1}" == 1 ]]; then
  timeout 240s srun --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --gpus-per-node=4 --gpu-bind=none \
    cmake --build csrc/cmake-build-release --target testFlashMoE --parallel 8
fi
echo "Starting 4-GPU FlashMoE diagnostic; job=$SLURM_JOB_ID"
echo "NVSHMEM_REMOTE_TRANSPORT=$NVSHMEM_REMOTE_TRANSPORT NVSHMEM_BOOTSTRAP=$NVSHMEM_BOOTSTRAP"
timeout 180s bash project/scripts/expert_granularity/launch_single_node.sh 4 \
  csrc/cmake-build-release/testFlashMoE 4096 2048 1408 16 6 1 1 1
echo "FlashMoE diagnostic completed"
