#!/bin/bash
# Source inside a Perlmutter GPU allocation. No dependency installation.
module load PrgEnv-gnu cudatoolkit/12.9 cmake nvshmem/3.2.5-1
export CUDACXX="${CUDA_HOME:?CUDA_HOME is unset}/bin/nvcc"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export NVSHMEM_LIB_HOME="${NVSHMEM_LIB_HOME:-/global/common/software/nersc9/nvshmem/3.2.5-1-25.03/lib}"
# Set this to YOUR installation, or explicitly select a shared installation.
export MATHDX_ROOT="${MATHDX_ROOT:-$HOME/.local/mathdx/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12}"
if [[ ! -d "$MATHDX_ROOT" ]]; then
  echo "Set MATHDX_ROOT to an existing MathDx installation before submitting." >&2
  return 1
fi
export CMAKE_PREFIX_PATH="${NVSHMEM_HOME:-}:$NVSHMEM_LIB_HOME:$MATHDX_ROOT:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$NVSHMEM_LIB_HOME:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP=MPI SLURM_MPI_TYPE=cray_shasta
# Single-node NVLink/CUDA P2P only. "none" disables off-node transport, not P2P.
# Keep all allocated peer GPUs visible; use launch_single_node.sh for 4 ranks.
export MPICH_GPU_SUPPORT_ENABLED=0 NVSHMEM_REMOTE_TRANSPORT=none
export NVSHMEM_DISABLE_GDRCOPY=1 NVSHMEM_DISABLE_NCCL=1
