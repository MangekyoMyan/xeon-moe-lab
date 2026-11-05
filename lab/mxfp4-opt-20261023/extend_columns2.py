#!/usr/bin/env python3
"""Compare two-input sharing and a four-plus-two remainder path."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
LAB = WORK.parent
ROOT = LAB / 'review-mxfp4-cols4-20261023'
BASE = LAB / 'review-mxfp4-packed-20261023/source'
GUARD = '#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'


def main():
    source = ROOT / 'source'
    shutil.copy2(ROOT / 'columns.patch', ROOT / 'columns4-stage.patch')
    function = (WORK / 'packed-cols4-kernel.c').read_text()
    function = function.replace('packed_cols4', 'packed_cols2')
    function = function.replace('const void * const vy[4]', 'const void * const vy[2]')
    function = function.replace('const block_q8_0 * y[4] = { vy[0], vy[1], vy[2], vy[3] };', 'const block_q8_0 * y[2] = { vy[0], vy[1] };')
    function = function.replace('__m256 acc[4];', '__m256 acc[2];').replace('c < 4', 'c < 2')
    (WORK / 'packed-cols2-kernel.c').write_text(function)
    rel = Path('ggml/src/ggml-cpu')
    qfile = source / rel / 'arch/x86/quants.c'
    assert 'packed_cols2(' not in qfile.read_text()
    qfile.write_text(qfile.read_text() + '\n' + GUARD + function + '\n#endif\n')
    header = source / rel / 'quants.h'
    code = header.read_text()
    anchor = (WORK / 'packed-cols4-kernel.c').read_text().splitlines()[0].replace(' {', ';')
    assert code.count(anchor) == 1
    header.write_text(code.replace(anchor, anchor + '\n' + function.splitlines()[0].replace(' {', ';')))
    path = source / rel / 'repack.cpp'
    code = path.read_text()
    start = code.index('    void forward_mul_mat_id(')
    end = code.index('\n    int repack(', start)
    segment = code[start:end]
    anchor = '''                if (cols4) {
                    for (; ir1 + 3 < nr1; ir1 += 4) {'''
    replacement = '''                static const bool cols2 = [] {
                    const char * env = std::getenv("GGML_CPU_MXFP4_COLS2");
                    return env != nullptr && std::strcmp(env, "1") == 0;
                }();
                if (cols4 || cols2) {
                    while (ir1 < nr1) {
                        const int ncols = cols4 && nr1 - ir1 >= 4 ? 4 : (cols2 && nr1 - ir1 >= 2 ? 2 : 0);
                        if (ncols == 0) break;'''
    assert segment.count(anchor) == 1
    segment = segment.replace(anchor, replacement).replace('c < 4', 'c < ncols')
    anchor = '                            ggml_vec_dot_mxfp4_q8_0_packed_cols4(ne00, sums, src0_cur + (row / 4) * 4 * nb01, row % 4, columns);'
    replacement = '''                            if (ncols == 4) {
                                ggml_vec_dot_mxfp4_q8_0_packed_cols4(ne00, sums, src0_cur + (row / 4) * 4 * nb01, row % 4, columns);
                            } else {
                                ggml_vec_dot_mxfp4_q8_0_packed_cols2(ne00, sums, src0_cur + (row / 4) * 4 * nb01, row % 4, columns);
                            }'''
    assert segment.count(anchor) == 1
    segment = segment.replace(anchor, replacement)
    anchor = '                            for (int c = 0; c < ncols; ++c) outputs[c][row] = sums[c];\n                        }'
    assert segment.count(anchor) == 1
    segment = segment.replace(anchor, anchor + '\n                        ir1 += ncols;')
    path.write_text(code[:start] + segment + code[end:])
    patch = []
    for name in (rel / 'arch/x86/quants.c', rel / 'quants.h', rel / 'repack.cpp'):
        patch += difflib.unified_diff((BASE / name).read_text().splitlines(True), (source / name).read_text().splitlines(True), fromfile='a/' + str(name), tofile='b/' + str(name))
    (ROOT / 'columns.patch').write_text(''.join(patch))
    build = json.loads((ROOT / 'build-commands.json').read_text())[1]
    with (ROOT / 'build-columns2.log').open('w') as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Two-input sharing build complete', flush=True)


if __name__ == '__main__':
    try:
        with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [ROOT, *ROOT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
