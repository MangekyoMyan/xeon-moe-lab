#!/usr/bin/env bash
set -euo pipefail

# CPU configuration measured on the dual E5-2630L v2 server.
for variable in ${!GGML_@} ${!LLAMA_ARG_@} ${!OMP_@} ${!GOMP_@}; do
  unset "$variable"
done
unset LD_LIBRARY_PATH
export LC_ALL=C
export GGML_CPU_NUMA_FULL=0
export GGML_CPU_NUMA_REPLICA=0
export GGML_CPU_MXFP4_ROWS=0
export GGML_CPU_MXFP4_PAIR=0
export GGML_CPU_MXFP4_PACKED=1
export GGML_CPU_MXFP4_COLS4=1
export GGML_CPU_MXFP4_PARALLEL_PACK=1
export GGML_CPU_REPACK_REUSE_BUFFER=1
export OMP_DYNAMIC=FALSE
export OMP_THREAD_LIMIT=24
export OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1

exec python3 /home/codex/projects/llm-lab/without_thp.py \
  numactl --physcpubind=0-23 --interleave=0,1 \
  /home/codex/projects/llm-lab/review-mxfp4-final-20261023/build/bin/llama-completion \
  -m /home/codex/projects/llm-lab/models/ggml-org--gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf \
  -t 24 -tb 24 -c 4096 -b 128 -ub 128 -n 1024 \
  --cpu-mask 0xffffff --cpu-strict 1 \
  --cpu-mask-batch 0xffffff --cpu-strict-batch 1 \
  --load-mode none --lazy-mode off \
  -ngl 0 --device none --no-kv-offload --no-op-offload \
  --jinja --reasoning auto --offline --simple-io \
  --color off --no-display-prompt --perf "$@"
