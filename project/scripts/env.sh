module load PrgEnv-gnu cudatoolkit/12.9 cmake nvshmem/3.2.5-1
module swap cudatoolkit cudatoolkit/12.9 2>/dev/null
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$HOME/.local/bin:$PATH
export NVSHMEM_LIB_HOME=/global/common/software/nersc9/nvshmem/3.2.5-1-25.03/lib
export MATHDX_ROOT=/global/homes/j/jbr266/.local/mathdx/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12
export CMAKE_PREFIX_PATH=$NVSHMEM_HOME:$NVSHMEM_LIB_HOME:$MATHDX_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$NVSHMEM_LIB_HOME:$LD_LIBRARY_PATH
export NVSHMEM_BOOTSTRAP=MPI
export SLURM_MPI_TYPE=cray_shasta
export MPICH_GPU_SUPPORT_ENABLED=0
