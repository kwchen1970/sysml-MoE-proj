#!/bin/bash
# Keep the allocated set visible for NVSHMEM P2P; never overwrite CUDA_VISIBLE_DEVICES.
set -euo pipefail
NGPU="${1:?usage: launch_single_node.sh <1|4> <program> [args...]}"
shift
if [[ $# -eq 0 ]]; then echo "Missing program" >&2; exit 2; fi
case "$NGPU" in
  1) exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1 "$@" ;;
  4) exec srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 "$@" ;;
  *) echo "Only single-node 1- and 4-GPU steps are supported." >&2; exit 2 ;;
esac
