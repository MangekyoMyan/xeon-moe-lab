#!/usr/bin/env bash
set -euo pipefail
# Experimental single-copy runner with 4 KiB NUMA interleave.
export GGML_CPU_NUMA_FULL=2
export GGML_CPU_NUMA_REPLICA=0
export GGML_CPU_NUMA_REPLICA_TRACE=0
export OMP_DYNAMIC=FALSE
export OMP_THREAD_LIMIT=24
export OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1
exec python3 /home/codex/projects/llm-lab/without_thp.py numactl --physcpubind=0-23 --interleave=0,1 \
  /home/codex/projects/llm-lab/review-full-replica-20261020/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/mradermacher--Melody1437-26B-A4B-i1-GGUF/Melody1437-26B-A4B.i1-Q6_K.gguf \
  -t 24 -tb 24 --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --load-mode mmap -ngl 0 -c 4096 -b 512 -ub 512 --jinja --offline "$@"
