#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$(realpath "${1:?three-arm work directory}")
src="$run/src-current"
flags=(-std=c++17 -O2 -march=ivybridge -mtune=ivybridge -mno-avx2 -mno-fma -fno-fast-math -ffp-contract=off)
"${CXX:-g++}" "${flags[@]}" -I"$src/ggml/include" -I"$src/ggml/src" -I"$src/xeon-local" \
    "$here/graph_probe.cpp" -L"$run/build-current/bin" -lggml-cpu -lggml-base -ldl -o "$run/graph_probe"
"${CXX:-g++}" "${flags[@]}" -shared -fPIC -I"$src/ggml/include" \
    "$here/count_paths.cpp" -ldl -o "$run/count_paths.so"
