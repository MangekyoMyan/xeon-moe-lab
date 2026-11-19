#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(realpath "${1:?path to a fixture output from reproduce.sh}")
"${CXX:-g++}" -O2 -std=c++17 -shared -fPIC -I "$out/src-base/ggml/include" "$here/count_paths.cpp" -ldl -o "$out/count_paths.so"
for v in base mixed; do
    for mask in 0 1 21 31 63; do
        LD_PRELOAD="$out/count_paths.so" XEON_TEST_COUNTS="$out/raw/paths-$v-$mask.csv" XEON_TEST_MASK="$mask" \
            LD_LIBRARY_PATH="$out/build-$v/bin" "$out/graph_probe" rounding 0 "$out/instrumented-$v-$mask.bin" > /dev/null
        cmp "$out/instrumented-$v-$mask.bin" "$out/$mask-0-base.bin"
    done
done
awk -F, 'NR==2 && !($1>0 && $2>0 && $3>0) {exit 1}' "$out/raw/paths-mixed-31.csv"
awk -F, 'NR==2 && !($1>0 && $3==0) {exit 1}' "$out/raw/paths-mixed-0.csv"
awk -F, 'NR==2 && !($1==0 && $3>0) {exit 1}' "$out/raw/paths-mixed-63.csv"
