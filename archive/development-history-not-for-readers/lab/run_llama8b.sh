#!/usr/bin/env bash
set -euo pipefail

# Decode profile validated on the current 2-socket E5-2630L v2 server.
# Extra arguments can override context size, sampling, or prompt settings.
export OPENBLAS_NUM_THREADS=1
exec numactl --physcpubind=0-23 --interleave=0,1 \
  /opt/llama.cpp/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/meta-llama-3.1-8b-instruct-q4_0.gguf \
  --numa numactl --load-mode none -t 24 -ngl 0 -c 4096 --offline "$@"
