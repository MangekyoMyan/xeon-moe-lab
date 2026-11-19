#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(realpath "${1:?source repository}")
out=$(realpath -m "${2:?fresh output directory}")
jobs=${3:-3}
if test -e "$out"; then echo "Use a new output directory" >&2; exit 1; fi
mkdir -p "$out/raw"
for v in base mixed; do
    src="$out/src-$v"
    python3 "$repo/scripts/prepare_source.py" --destination "$src"
    python3 "$here/prepare.py" "$src" "$v"
    "$here/../build_cpu.sh" "$src" "$out/build-$v" "$jobs" > "$out/raw/build-$v.log" 2>&1
done
"${CXX:-g++}" -O2 -std=c++17 -march=ivybridge -mno-avx2 -mno-fma -ffp-contract=off -fno-fast-math \
    -I "$out/src-base/ggml/include" -I "$out/src-base/ggml/src" -I "$out/src-base/xeon-local" \
    "$here/../graph_probe.cpp" -L "$out/build-base/bin" -lggml-cpu -lggml-base -o "$out/graph_probe"
for mask in 0 1 21 31 63; do
    for mode in 0 1 2 3; do
        for v in base mixed; do
            XEON_TEST_MASK="$mask" LD_LIBRARY_PATH="$out/build-$v/bin" "$out/graph_probe" rounding "$mode" "$out/$mask-$mode-$v.bin" > "$out/raw/$mask-$mode-$v.csv"
        done
        cmp "$out/$mask-$mode-base.bin" "$out/$mask-$mode-mixed.bin"
    done
done
(cd "$out" && sha256sum *.bin build-*/bin/libggml-cpu.so) > "$out/raw/hashes.txt"
"$here/check_paths.sh" "$out"
echo "Mixed dispatch comparison passed."
