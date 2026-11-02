#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/j/jbr266/moe/FlashMoE/project/results/raw/sweep_%j.log
source $SCRATCH/moe/FlashMoE/project/scripts/env.sh
BIN=$SCRATCH/moe/FlashMoE/csrc/cmake-build-release/testFlashMoE
run () {  # ngpu tokens H I E k
  for rep in 1 2 3; do
    if [ $1 -eq 1 ]; then
      srun -n 1 --gpus=1 $BIN $2 $3 $4 $5 $6 2>&1 | grep -E "^[0-9]+,"
    else
      srun -n $1 --gpus-per-node=4 $BIN $2 $3 $4 $5 $6 2>&1 | grep -E "^[0-9]+,"
    fi
  done
}
echo "rank, dtype, S, H, I, E, k, FlashMoE_Time(ms), error(%), EC, GPU, SM, Topology, MLPType, bM, bN0, bK0, bN1, bK1, threads, blocks/SM, SMs, blocks, rtol, atol, graph_launches, warmup, runs, workspace(MiB)"
for NG in 1 4; do
  for T in 1024 2048 4096 8192 16384; do run $NG $T 2048 1408 64 6; done
  for E in 16 32 64 128;             do run $NG 4096 2048 1408 $E 6; done
  for I in 1408 4096 14336;          do run $NG 4096 2048 $I 16 2; done
done
