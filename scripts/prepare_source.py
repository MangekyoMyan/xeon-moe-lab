#!/usr/bin/env python3
"""Prepare a source tree from the pinned llama.cpp fork and the selected overlays."""
from pathlib import Path
import argparse,hashlib,json,shutil,tempfile
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

CURRENT_GLUE = GLUE + '''
target_sources(ggml-cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/mx-rhs.cpp")
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.c"
    TARGET_DIRECTORY ggml-cpu APPEND PROPERTY COMPILE_DEFINITIONS XEON_MX_PREPACK=1)
set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/xeon-local/mx-rhs.cpp"
    TARGET_DIRECTORY ggml-cpu APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
'''

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['baseline', 'current'], default='baseline')
    p.add_argument('--destination', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    profile_dir = ROOT / ('current' if a.profile == 'current' else 'accepted')
    dest = (a.destination or ROOT / 'work' / (profile_dir.name + '-source')).resolve()
    vendor = ROOT / 'vendor/llama.cpp'
    glue = CURRENT_GLUE if a.profile == 'current' else GLUE
    if dest == ROOT or dest.is_relative_to(vendor) or dest.is_relative_to(profile_dir):
        raise SystemExit('Choose a destination outside the archived source and overlays')
    origins=json.loads((profile_dir/'ORIGINS.json').read_text())
    for row in origins['files']:
        if hashlib.sha256((ROOT/row['archive']).read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Overlay hash mismatch: '+row['archive'])
    if not a.verify:
        if dest.exists():raise SystemExit('Destination exists; use --verify or choose a new directory')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
            stage = Path(scratch) / 'source'
            shutil.copytree(vendor, stage)
            for row in origins['files']:
                out = stage / row['target_in_source']
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / row['archive'], out)
            with (stage / 'CMakeLists.txt').open('a') as f:
                f.write(glue)
            stage.rename(dest)
    for row in origins['files']:
        if hashlib.sha256((dest/row['target_in_source']).read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Prepared source mismatch')
    if (dest/'CMakeLists.txt').read_text()!=(vendor/'CMakeLists.txt').read_text()+glue:raise SystemExit('Build glue mismatch')
    print(f'{a.profile} source tree verified at {dest}')
if __name__=='__main__':main()
