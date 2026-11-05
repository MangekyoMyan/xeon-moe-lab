#!/usr/bin/env python3
"""Reuse each packed MXFP4 row across four prefill input vectors."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

WORK = Path(__file__).resolve().parent
LAB = WORK.parent
BASE = LAB / 'review-mxfp4-packed-20261023'
ROOT = LAB / 'review-mxfp4-cols4-20261023'
GUARD = '#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'


def main():
    ROOT.mkdir(exist_ok=False)
    shutil.copytree(BASE / 'source', ROOT / 'source')
    source = ROOT / 'source'
    rel = Path('ggml/src/ggml-cpu')
    function = (WORK / 'packed-cols4-kernel.c').read_text()
    qfile = source / rel / 'arch/x86/quants.c'
    qfile.write_text(qfile.read_text() + '\n' + GUARD + function + '\n#endif\n')
    header = source / rel / 'quants.h'
    code = header.read_text()
    anchor = 'void ggml_vec_dot_mxfp4_q8_0_rows4_packed(int n, float * s, const void * vx, const void * vy);'
    assert code.count(anchor) == 1
    header.write_text(code.replace(anchor, anchor + '\n' + function.splitlines()[0].replace(' {', ';')))
    path = source / rel / 'repack.cpp'
    code = path.read_text()
    start = code.index('    void forward_mul_mat_id(')
    end = code.index('\n    int repack(', start)
    segment = code[start:end]
    anchor = '            for (int ir1 = 0; ir1 < nr1; ir1++) {'
    assert segment.count(anchor) == 1
    replacement = '''            int64_t ir1 = 0;
''' + GUARD + '''            if constexpr (INTER_SIZE == 16) {
                static const bool cols4 = [] {
                    const char * env = std::getenv("GGML_CPU_MXFP4_COLS4");
                    return env != nullptr && std::strcmp(env, "1") == 0;
                }();
                if (cols4) {
                    for (; ir1 + 3 < nr1; ir1 += 4) {
                        const void * columns[4];
                        float * outputs[4];
                        for (int c = 0; c < 4; ++c) {
                            const mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, ir1 + c);
                            columns[c] = wdata + (mapping.i1 % ne11) * nbw1 + mapping.i2 * nbw2;
                            outputs[c] = (float *) ((char *) dst->data + mapping.i1 * nb1 + mapping.i2 * nb2);
                        }
                        for (int64_t row = src0_cur_start; row < src0_cur_end; ++row) {
                            float sums[4];
                            ggml_vec_dot_mxfp4_q8_0_packed_cols4(ne00, sums, src0_cur + (row / 4) * 4 * nb01, row % 4, columns);
                            for (int c = 0; c < 4; ++c) outputs[c][row] = sums[c];
                        }
                    }
                }
            }
#endif
            for (; ir1 < nr1; ir1++) {'''
    segment = segment.replace(anchor, replacement)
    path.write_text(code[:start] + segment + code[end:])
    patch = []
    for name in (rel / 'arch/x86/quants.c', rel / 'quants.h', rel / 'repack.cpp'):
        patch += difflib.unified_diff((BASE / 'source' / name).read_text().splitlines(True), (source / name).read_text().splitlines(True), fromfile='a/' + str(name), tofile='b/' + str(name))
    (ROOT / 'columns.patch').write_text(''.join(patch))
    commands = [[x.replace(str(BASE), str(ROOT)) for x in command] for command in json.loads((BASE / 'build-commands.json').read_text())]
    (ROOT / 'build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for label, command in zip(('configure', 'build'), commands):
        print(label + ' column-sharing runtime', flush=True)
        with (ROOT / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Column-sharing runtime complete', flush=True)


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
