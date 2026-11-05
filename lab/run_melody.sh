#!/usr/bin/env bash
set -euo pipefail

# Decode profile validated on the current 2-socket E5-2630L v2 server.
export OPENBLAS_NUM_THREADS=1
exec numactl --physcpubind=0-23 --interleave=0,1 \
  /opt/llama.cpp/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/mradermacher--Melody1437-26B-A4B-i1-GGUF/Melody1437-26B-A4B.i1-Q6_K.gguf \
  --numa numactl --load-mode none -t 24 -ngl 0 -c 4096 --jinja --offline "$@"
