#!/usr/bin/env python3
"""Keep the selected four-input path and parallelize lossless weight packing."""
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
LAB = WORK.parent
BASE = LAB / 'review-mxfp4-packed-20261023'
COLUMNS = LAB / 'review-mxfp4-cols4-20261023'
ROOT = LAB / 'review-mxfp4-final-20261023'


def main():
    ROOT.mkdir(exist_ok=False)
    shutil.copytree(BASE / 'source', ROOT / 'source')
    source = ROOT / 'source'
    command = ['patch', '--batch', '-s', '-p1', '-d', str(source), '-i', str(COLUMNS / 'columns4-stage.patch')]
    subprocess.run(command, check=True)
    prior = json.loads((COLUMNS / 'columns4-stage-hashes.json').read_text())
    for path, expected in prior.items():
        if '/source/' in path:
            current = source / path.split('/source/', 1)[1]
            assert hashlib.sha256(current.read_bytes()).hexdigest() == expected
    path = source / 'ggml/src/ggml-cpu/repack.cpp'
    original = path.read_text()
    (ROOT / 'before-parallel-repack.cpp').write_text(original)
    code = original.replace('#include <cstdlib>', '#include <cstdlib>\n#ifdef GGML_USE_OPENMP\n#include <omp.h>\n#endif')
    start = code.index('static int repack_mxfp4_to_mxfp4_4_bl(')
    end = code.index('static block_mxfp4x8 make_block_mxfp4x8(', start)
    function = '''static int repack_mxfp4_to_mxfp4_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 16);
    GGML_ASSERT(ggml_nrows(t) <= INT32_MAX && t->ne[0] >= 0 && t->ne[0] / QK_MXFP4 <= INT32_MAX);
    const block_mxfp4 * src = (const block_mxfp4 *) data;
    block_mxfp4x4 * dst = (block_mxfp4x4 *) t->data;
    const int nrow = ggml_nrows(t);
    const int nblocks = t->ne[0] / QK_MXFP4;
    GGML_ASSERT(data_size == (size_t) nrow * nblocks * sizeof(block_mxfp4));
    if (t->ne[1] % 4 != 0 || t->ne[0] % QK_MXFP4 != 0) return -1;

#ifdef GGML_USE_OPENMP
    const char * env = std::getenv("GGML_CPU_MXFP4_PARALLEL_PACK");
    const bool parallel = interleave_block == 16 && data_size >= 16 * 1024 * 1024 &&
                          env != nullptr && std::strcmp(env, "1") == 0 && !omp_in_parallel();
#pragma omp parallel for schedule(static) if(parallel)
#endif
    for (int row = 0; row < nrow; row += 4) {
        const block_mxfp4 * src_group = src + (size_t) row * nblocks;
        block_mxfp4x4 * dst_group = dst + (size_t) (row / 4) * nblocks;
        if (interleave_block == 16) {
            for (int block = 0; block < nblocks; ++block) {
                for (int r = 0; r < 4; ++r) {
                    const block_mxfp4 & in = src_group[block + (size_t) r * nblocks];
                    dst_group[block].e[r] = in.e;
                    memcpy(dst_group[block].qs + r * 16, in.qs, 16);
                }
            }
        } else {
            block_mxfp4 tmp[4];
            for (int block = 0; block < nblocks; ++block) {
                for (int r = 0; r < 4; ++r) tmp[r] = src_group[block + (size_t) r * nblocks];
                dst_group[block] = make_block_mxfp4x4(tmp, interleave_block);
            }
        }
    }
    return 0;
}

'''
    path.write_text(code[:start] + function + code[end:])
    (ROOT / 'parallel-pack.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), path.read_text().splitlines(True), fromfile='a/ggml/src/ggml-cpu/repack.cpp', tofile='b/ggml/src/ggml-cpu/repack.cpp')))
    commands = [[x.replace(str(BASE), str(ROOT)) for x in row] for row in json.loads((BASE / 'build-commands.json').read_text())]
    (ROOT / 'build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for label, command in zip(('configure', 'build'), commands):
        print(label + ' final runtime', flush=True)
        with (ROOT / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Final runtime complete', flush=True)


if __name__ == '__main__':
    try:
        with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    finally:
        if os.geteuid() == 0 and ROOT.exists():
            owner = LAB.stat()
            for path in [ROOT, *ROOT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
