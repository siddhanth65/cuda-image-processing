#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

make build
./cuda_image_processing \
  --input "${1:-data/sample_images}" \
  --output "${2:-output}" \
  --results "${3:-results/benchmark.csv}" \
  --operation "${4:-all}" \
  --batch-size "${5:-16}"
