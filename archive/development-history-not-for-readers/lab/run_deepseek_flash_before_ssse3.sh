#!/usr/bin/env bash
set -euo pipefail

# Measured CPU profile: OpenMP on, BLAS off, one thread per physical core.
# Keep file-backed weights cached instead of making an anonymous copy.
exec numactl --physcpubind=0-23 --interleave=0,1 \
  /home/codex/projects/llm-lab/build-native-noblas/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf \
  --load-mode mmap -t 12 -tb 12 -ngl 0 -c 4096 -b 256 -ub 256 \
  --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --jinja --offline "$@"
