#!/usr/bin/env bash
# Build every comparison in its own reconstructed source tree and build directory.
set -euo pipefail
repo=$(realpath "${1:?source repository containing scripts/prepare_source.py}")
out=$(realpath -m "${2:?new output directory}")
jobs=${3:-3}
here=$(cd "$(dirname "$0")" && pwd)
if test -e "$out"; then echo "Use a new output directory: $out" >&2; exit 1; fi
mkdir -p "$out/raw"
variants=(baseline route2 q8 iqsafe combosafe)
declare -A diffs=([route]=01-route-workspace.patch [iq]=02-compact-input-signs.patch [route2]=03-route-selected.patch [q8]=04-q8k-avx1.patch [combined]=05-combined.patch [iqsafe]=06-compact-signed-byte.patch [combosafe]=07-combined-corrected.patch)
if test "${INCLUDE_ARCHIVED:-0}" = 1; then variants+=(route iq combined); fi
for v in "${variants[@]}"; do
    src="$out/src-$v"
    python3 "$repo/scripts/prepare_source.py" --destination "$src"
    python3 "$repo/scripts/prepare_source.py" --destination "$src" --verify
    if test "$v" != baseline; then
        patch --batch --fuzz=0 -p1 -d "$src" < "$here/${diffs[$v]}" > "$out/raw/apply-$v.log"
    fi
    "$here/build_cpu.sh" "$src" "$out/build-$v" "$jobs" > "$out/raw/build-$v.log" 2>&1
    ln -s "$out/build-$v/bin" "$out/lib-$v"
done
src=$out/src-baseline
flags=(-O2 -std=c++17 -march=ivybridge -mno-avx2 -mno-fma -ffp-contract=off -fno-fast-math)
includes=(-I "$src/ggml/include" -I "$src/ggml/src" -I "$src/xeon-local")
"${CXX:-g++}" "${flags[@]}" "${includes[@]}" "$here/graph_probe.cpp" -L "$out/lib-baseline" -lggml-cpu -lggml-base -o "$out/graph_probe"
"${CXX:-g++}" "${flags[@]}" -frounding-math "${includes[@]}" "$here/q8_probe.cpp" -L "$out/lib-baseline" -lggml-cpu -lggml-base -o "$out/q8_probe"
"${CXX:-g++}" "${flags[@]}" "${includes[@]}" "$here/parent_rounding_probe.cpp" -ldl -o "$out/parent_rounding"
for v in "${variants[@]}"; do
    LD_LIBRARY_PATH="$out/lib-$v" "$out/graph_probe" plans > "$out/raw/plans-$v.csv"
    for m in check edges; do
        LD_LIBRARY_PATH="$out/lib-$v" "$out/graph_probe" "$m" "$out/$m-$v.bin" > "$out/raw/$m-$v.csv"
        cmp "$out/$m-baseline.bin" "$out/$m-$v.bin"
    done
    for rm in 0 1 2 3; do
        LD_LIBRARY_PATH="$out/lib-$v" "$out/graph_probe" rounding "$rm" "$out/round-$v-$rm.bin" > "$out/raw/round-$v-$rm.csv"
        if [[ "$v" == iq || "$v" == combined ]] && test "$rm" = 1; then
            if cmp -s "$out/round-baseline-$rm.bin" "$out/round-$v-$rm.bin"; then echo "Archived FE_DOWNWARD mismatch was absent" >&2; exit 1; fi
        else
            cmp "$out/round-baseline-$rm.bin" "$out/round-$v-$rm.bin"
        fi
    done
done
for v in iqsafe combosafe; do
    LD_LIBRARY_PATH="$out/lib-$v" "$out/parent_rounding" "$out/lib-$v/libggml-cpu.so" > "$out/raw/parent-rounding-$v.csv"
    awk -F, 'NR>1 && $10 != 1 {exit 1}' "$out/raw/parent-rounding-$v.csv"
done
LD_LIBRARY_PATH="$out/lib-q8" "$out/q8_probe" check > "$out/raw/q8-byte-check.csv"
for mode in disabled forced; do
    if test "$mode" = disabled; then on=0; force=0; else on=1; force=1; fi
    for v in baseline route2; do
        GGML_CPU_TILED_MM="$on" GGML_CPU_TILED_MM_FORCE="$force" LD_LIBRARY_PATH="$out/lib-$v" \
            "$out/graph_probe" edges "$out/edges-$mode-$v.bin" > "$out/raw/edges-$mode-$v.csv"
        cmp "$out/edges-$mode-baseline.bin" "$out/edges-$mode-$v.bin"
    done
done
(cd "$out" && sha256sum *.bin lib-*/libggml-cpu.so) > "$out/raw/hashes.txt"
echo "Byte checks passed. Outputs: $out"
