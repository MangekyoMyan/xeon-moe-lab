#!/usr/bin/env bash
set -euo pipefail

# Private source: upstream 436f6f89e plus completion-template-kwargs and ssse3-iq-sign-unpack patches.
cmake -S /home/codex/projects/llm-lab/llama-private \
  -B /home/codex/projects/llm-lab/build-avx-unpack \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON -DGGML_BLAS=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build /home/codex/projects/llm-lab/build-avx-unpack \
  --target llama-bench llama-completion test-backend-ops test-quantize-fns -j 12
