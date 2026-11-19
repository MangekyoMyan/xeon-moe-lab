#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
accepted=$(realpath "${1:?accepted source repository}")
archive=$(realpath "${2:?git checkout containing 37f6390}")
run=$(realpath -m "${3:?new work directory}")
jobs=${4:-3}
mkdir -p "$run/raw"
for variant in current 08 09; do
    src="$run/src-$variant"
    build="$run/build-$variant"
    test ! -e "$src" && test ! -e "$build" || { echo "Choose a fresh source/build directory." >&2; exit 1; }
    python3 "$here/prepare_variant.py" "$accepted" "$archive" "$src" "$variant" > "$run/raw/sources-$variant.log" 2>&1
    bash "$here/build_cpu.sh" "$src" "$build" "$jobs" > "$run/raw/build-$variant.log" 2>&1
    mkdir "$run/lib-$variant"
    ln -s "$build/bin/libggml-cpu.so.0" "$run/lib-$variant/libggml-cpu.so.0"
    ln -s "$run/build-current/bin/libggml-base.so.0" "$run/lib-$variant/libggml-base.so.0"
done
