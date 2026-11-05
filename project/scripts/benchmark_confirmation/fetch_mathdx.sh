#!/bin/bash
# Download MathDx 25.12.0 (CUDA 12), which provides cuBLASDx and cuRANDDx.
# Unpack layout matches MATHDX_ROOT in env.sh: $SCRATCH/mathdx/nvidia/mathdx/25.12
set -euo pipefail

DEST_PARENT="${SCRATCH}/mathdx"
MARKER="${DEST_PARENT}/nvidia-mathdx-25.12.0-cuda12/nvidia/mathdx/25.12/include/cublasdx.hpp"
if [[ -f "$MARKER" ]]; then
  echo "MathDx already present at ${DEST_PARENT}"
  exit 0
fi

mkdir -p "${SCRATCH}/src"
TARBALL="${SCRATCH}/src/nvidia-mathdx-25.12.0-cuda12.tar.gz"
URL="https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/cuda12/nvidia-mathdx-25.12.0-cuda12.tar.gz"
if [[ ! -s "$TARBALL" ]]; then
  wget -O "$TARBALL" "$URL"
fi
mkdir -p "$DEST_PARENT"
tar -xzf "$TARBALL" -C "$DEST_PARENT"
test -f "$MARKER"
echo "Installed MathDx under ${DEST_PARENT}"
