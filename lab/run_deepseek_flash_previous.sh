#!/usr/bin/env bash
set -euo pipefail

# Measured CPU-only profile for the existing 73 GiB IQ2_XXS-XL model.
# Create BLAS workers before GGML pins the main thread to CPU 0.
export OPENBLAS_NUM_THREADS=24
exec numactl --physcpubind=0-23 --interleave=0,1 \
  /opt/llama.cpp/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf \
  --load-mode none -t 24 -tb 24 -ngl 0 -c 4096 -b 256 -ub 128 \
  --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --jinja --offline "$@"
