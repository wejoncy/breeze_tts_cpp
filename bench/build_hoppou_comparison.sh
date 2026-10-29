#!/usr/bin/env bash
set -euo pipefail
baseline=${1:?usage: build_hoppou_comparison.sh BASELINE_REPO [BASELINE_BUILD_DIR]}
baseline=$(cd -- "$baseline" && pwd)
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cmake=${CMAKE:-cmake}
"$cmake" -S "$project" -B "$project/build" \
  -DTTS_HOPPOU_SOURCE_DIR="$baseline" -DTTS_HOPPOU_BUILD_DIR="${2:-$baseline/build-cuda}"
"$cmake" --build "$project/build" --config Release --target compare_hoppou
