#!/usr/bin/env bash
set -euo pipefail

# Measured CPU profile: OpenMP on, BLAS off, SSSE3 IQ unpack, paired MoE, Q8 rows4 and bounded Q8 prefetch.
# Optional: GGML_CPU_NUMA_REPLICA=1 uses about 6 GiB extra RSS for a small generation gain.
# Use 24 workers for long prompts and 12 for generation; -tb 12 favors short prompts.
# Keep file-backed weights cached instead of making an anonymous copy.
exec numactl --physcpubind=0-23 --interleave=0,1 \
  /home/codex/projects/llm-lab/build-avx-prefetch/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf \
  --load-mode mmap -t 12 -tb 24 -ngl 0 -c 4096 -b 512 -ub 512 \
  --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --jinja --offline "$@"
