#!/bin/bash
# Configure and build csrc/testFlashMoE on a Perlmutter login or GPU node.
# Login nodes have one A100, which is enough for CMake's native-arch probe.
# Both login and compute A100s are sm_80 with 108 SMs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
"${SCRIPT_DIR}/fetch_mathdx.sh"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc missing under CUDA_HOME=${CUDA_HOME}" >&2
  exit 1
fi
nvcc --version | tail -1

BUILD="${REPO_ROOT}/csrc/cmake-build-release"
cmake -DCMAKE_BUILD_TYPE=Release -Wno-dev -G Ninja -S "${REPO_ROOT}/csrc" -B "$BUILD"
cmake --build "$BUILD" --target testFlashMoE --parallel "${BUILD_JOBS:-8}"
test -x "${BUILD}/testFlashMoE"
echo "Built ${BUILD}/testFlashMoE"
