#!/usr/bin/env python3
"""Build a separate inference runtime for full-graph and model checks."""
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess

LAB = Path(__file__).resolve().parent.parent
ROOT = LAB / 'review-mxfp4-20261023'
BASE = LAB / 'review-glm-q8-columns4-20261020/source'


def main():
    ROOT.mkdir(exist_ok=False)
    shutil.copytree(BASE, ROOT / 'source')
    source = ROOT / 'source'
    qrel = Path('ggml/src/ggml-cpu/arch/x86/quants.c')
    crel = Path('ggml/src/ggml-cpu/ggml-cpu.c')
    shutil.copyfile(LAB / 'mxfp4-opt-20261023/narrow.c', source / qrel)
    path = source / crel
    code = path.read_text()
    old = 'static bool ggml_cpu_moe_pair_trace = false;'
    code = code.replace(old, old + '\nstatic bool ggml_cpu_mxfp4_pair = false;')
    old = '        a->src[0]->type != GGML_TYPE_IQ2_XXS || b->src[0]->type != GGML_TYPE_IQ2_XXS ||'
    new = '''        a->src[0]->type != b->src[0]->type ||
        !(a->src[0]->type == GGML_TYPE_IQ2_XXS || (ggml_cpu_mxfp4_pair && a->src[0]->type == GGML_TYPE_MXFP4)) ||'''
    assert code.count(old) == 1
    code = code.replace(old, new)
    old = '            env = getenv("GGML_CPU_MOE_PAIR_TRACE");'
    assert code.count(old) == 1
    code = code.replace(old, '            env = getenv("GGML_CPU_MXFP4_PAIR");\n            ggml_cpu_mxfp4_pair = (env != NULL && strcmp(env, "1") == 0);\n' + old)
    path.write_text(code)
    patch = []
    for rel in (qrel, crel):
        patch += difflib.unified_diff((BASE / rel).read_text().splitlines(True), (source / rel).read_text().splitlines(True), fromfile='a/' + str(rel), tofile='b/' + str(rel))
    (ROOT / 'candidate.patch').write_text(''.join(patch))
    configure = ['cmake', '-S', str(source), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DGGML_NATIVE=ON', '-DGGML_OPENMP=ON', '-DGGML_BLAS=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF', '-DGGML_OPENCL=OFF',
                 '-DGGML_METAL=OFF', '-DGGML_SYCL=OFF', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_TESTS=ON']
    build = ['cmake', '--build', str(ROOT / 'build'), '--target', 'llama-bench', 'llama-completion', 'test-backend-ops', 'test-quantize-fns', '-j', '8']
    (ROOT / 'build-commands.json').write_text(json.dumps([configure, build], indent=2) + '\n')
    for name, command in [('configure', configure), ('build', build)]:
        print(name, flush=True)
        with (ROOT / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('candidate build complete', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        owner = LAB.stat()
        if os.geteuid() == 0 and ROOT.exists():
            for path in [ROOT, *ROOT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
