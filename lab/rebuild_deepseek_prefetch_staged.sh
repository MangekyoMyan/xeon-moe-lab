#!/usr/bin/env bash
set -euo pipefail

# Private source: upstream 436f6f89e plus the locally verified CPU changes.
cmake -S /home/codex/projects/llm-lab/llama-private-prefetch \
  -B /home/codex/projects/llm-lab/build-avx-prefetch \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build /home/codex/projects/llm-lab/build-avx-prefetch \
  --target llama-bench llama-completion test-backend-ops test-quantize-fns -j 8
