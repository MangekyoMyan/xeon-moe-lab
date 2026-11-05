#!/usr/bin/env bash
set -euo pipefail

# GLM Q8 prefill shares each weight row across four input columns on AVX CPUs.
for variable in ${!GGML_@} ${!LLAMA_ARG_@} ${!OMP_@} ${!GOMP_@}; do
  unset "$variable"
done
unset LD_LIBRARY_PATH
export LC_ALL=C
export GGML_CPU_NUMA_FULL=0
export GGML_CPU_NUMA_REPLICA=0
export OMP_DYNAMIC=FALSE
export OMP_THREAD_LIMIT=24
export OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1

exec numactl --physcpubind=0-23 --interleave=0,1 \
  /home/codex/projects/llm-lab/review-glm-q8-columns4-20261020/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf \
  -t 24 -tb 24 -c 4096 -b 128 -ub 128 -n 256 \
  --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --load-mode mmap --lazy-mode off --no-repack \
  -ngl 0 --device none --no-kv-offload --no-op-offload \
  --jinja --reasoning off --temp 0 --seed 42 --offline \
  --single-turn --simple-io --color off --no-display-prompt --perf "$@"
