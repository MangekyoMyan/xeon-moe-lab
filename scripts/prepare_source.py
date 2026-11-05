#!/usr/bin/env python3
"""Reconstruct adopted source into a new ignored directory; do not build or start anything."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
GLUE='''
# Archival reconstruction glue: accepted source files are unchanged.
target_sources(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/iq3-cache.cpp")
target_include_directories(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local")
target_include_directories(llama PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local")
set_property(SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/iq3-cache.cpp"
    TARGET_DIRECTORY ggml-cpu APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/llama-model.cpp"
    TARGET_DIRECTORY llama APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
'''
def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=ROOT/'work/accepted-source');p.add_argument('--verify',action='store_true');a=p.parse_args();dest=a.destination.resolve()
    if dest==ROOT or dest==ROOT/'vendor/llama.cpp':raise SystemExit('Refusing source/archive overwrite')
    origins=json.loads((ROOT/'accepted/ORIGINS.json').read_text())
    for row in origins['files']:
        if hashlib.sha256((ROOT/row['archive']).read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Overlay hash mismatch: '+row['archive'])
    if not a.verify:
        if dest.exists():raise SystemExit('Destination exists; use --verify or choose a new directory')
        shutil.copytree(ROOT/'vendor/llama.cpp',dest)
        for row in origins['files']:
            out=dest/row['target_in_source'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/row['archive'],out)
        with (dest/'CMakeLists.txt').open('a') as f:f.write(GLUE)
    for row in origins['files']:
        if hashlib.sha256((dest/row['target_in_source']).read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Prepared source mismatch')
    if (dest/'CMakeLists.txt').read_text()!=(ROOT/'vendor/llama.cpp/CMakeLists.txt').read_text()+GLUE:raise SystemExit('Build glue mismatch')
    print('Accepted source reconstructed and checked:',dest)
    print('No compilation, model startup, or deployment performed.')
if __name__=='__main__':main()
