#!/usr/bin/env python3
"""Integrate equal-size AVX MXFP4 packing through the existing CPU_REPACK backend."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
from prepare_packed import packed_kernel

LAB = Path(__file__).resolve().parent.parent
ROOT = LAB / 'review-mxfp4-packed-20261023'
BASE = LAB / 'review-mxfp4-20261023/source'
GUARD = '#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'


def fix_mapping(code):
    start = code.index('template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits')
    end = code.index('}  // namespace ggml::cpu::repack', start)
    segment = code[start:end]
    old = '                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);'
    new = '''                    const int64_t n_ids = op->src[2]->ne[0];
                    GGML_ASSERT(n_ids > 0 && ne12 >= 0 && ne12 <= (INT64_MAX - 1) / n_ids);
                    const int64_t rows_per_expert = INTER_SIZE == 16 ? ne12 * n_ids : ne12;
                    GGML_ASSERT(ne02 > 0 && (uint64_t) (rows_per_expert + 1) <= (SIZE_MAX - size) / sizeof_mmid_row_mapping / (uint64_t) ne02);
                    size += sizeof_mmid_row_mapping * ne02 * (rows_per_expert + 1);'''
    assert segment.count(old) == 1
    segment = segment.replace(old, new)
    old = '        const int n_as  = ne02;       // n_expert'
    new = old + '''
        GGML_ASSERT(n_ids > 0 && ne12 >= 0 && ne12 <= (INT64_MAX - 1) / n_ids);
        const int64_t rows_per_expert = INTER_SIZE == 16 ? ne12 * n_ids : ne12;'''
    assert segment.count(old) == 1
    segment = segment.replace(old, new)
    segment = segment.replace('n_as*(ne12 + 1)*sizeof(mmid_row_mapping)', 'n_as * (rows_per_expert + 1) * sizeof(mmid_row_mapping)')
    segment = segment.replace('// total of [n_as][ne12 + 1] elements', '// total of [n_as][rows_per_expert + 1] elements')
    segment = segment.replace('// [n_as][ne12]', '// [n_as][rows_per_expert]')
    segment = segment.replace('matrix_rows[(row_id) * ne12 + (i1)]', 'matrix_rows[(row_id) * rows_per_expert + (i1)]')
    old = '                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };'
    assert segment.count(old) == 1
    segment = segment.replace(old, '                    GGML_ASSERT(matrix_row_counts[i02] < rows_per_expert);\n' + old)
    return code[:start] + segment + code[end:]


def main():
    ROOT.mkdir(exist_ok=False)
    shutil.copytree(BASE, ROOT / 'source')
    source = ROOT / 'source'
    rel = Path('ggml/src/ggml-cpu')
    qfile = source / rel / 'arch/x86/quants.c'
    code = qfile.read_text()
    function = packed_kernel()
    old = 'void ggml_vec_dot_mxfp4_q8_0_grouped(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc)'
    declaration = 'void ggml_vec_dot_mxfp4_q8_0_rows4_packed(int n, float * s, const void * vx, const void * vy)'
    function = function.replace(old, declaration)
    function = function.replace('    if (nrc == 1) { ggml_vec_dot_mxfp4_q8_0(n, s, bs, vx, bx, vy, by, nrc); return; }\n', '')
    function = function.replace('assert(nrc == 4 && n >= 0 && n % 32 == 0);', 'assert(n >= 0 && n % 32 == 0);')
    qfile.write_text(code + '\n' + GUARD + function + '#endif\n')
    header = source / rel / 'quants.h'
    code = header.read_text()
    anchor = 'void ggml_vec_dot_mxfp4_q8_0_rows4(int n, float * s, const void * vx, size_t bx, const void * vy);'
    assert code.count(anchor) == 1
    header.write_text(code.replace(anchor, anchor + '\n' + declaration + ';'))
    repack = source / rel / 'repack.cpp'
    code = repack.read_text().replace('#include "repack.h"', '#include "repack.h"\n#include "quants.h"\n#include <cstdlib>')
    anchor = 'template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>'
    position = code.index(anchor)
    # This internal format keeps activation rows in the normal Q8_0 layout.
    extra = GUARD + '''template <> void ggml_quantize_mat_t<16, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for (int64_t r = 0; r < nrow; ++r) {
        from_float(x + r * n_per_row, (char *) vy + r * stride, n_per_row);
    }
}
#endif

'''
    code = code[:position] + extra + code[position:]
    start = code.index('static block_mxfp4x4 make_block_mxfp4x4(')
    end = code.index('static block_mxfp4x8 make_block_mxfp4x8(', start)
    segment = code[start:end]
    segment = segment.replace('if (blck_size_interleave == 4)', 'if (blck_size_interleave == 4 || blck_size_interleave == 16)')
    segment = segment.replace('sizeof(uint32_t)', 'blck_size_interleave')
    segment = segment.replace('GGML_ASSERT(interleave_block == 4);', 'GGML_ASSERT(interleave_block == 4 || interleave_block == 16);')
    segment = segment.replace('data_size == nrow * nblocks * sizeof(block_mxfp4)', 'data_size == (size_t) nrow * nblocks * sizeof(block_mxfp4)')
    code = code[:start] + segment + code[end:]
    anchor = 'template <> int repack<block_mxfp4, 8, 8>'
    position = code.index(anchor)
    extra = GUARD + '''template <> int repack<block_mxfp4, 16, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_4_bl(t, 16, data, data_size);
}
#endif

'''
    code = code[:position] + extra + code[position:]
    anchor = 'template <> void gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>'
    position = code.index(anchor)
    extra = GUARD + '''template <> void gemv<block_mxfp4, 16, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(nr == 1 && n >= 0 && n % QK_MXFP4 == 0 && nc % 4 == 0);
    const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, n);
    for (int r = 0; r < nc; r += 4) {
        ggml_vec_dot_mxfp4_q8_0_rows4_packed(n, s + r, (const char *) vx + r * stride, vy);
    }
    GGML_UNUSED(bs);
}
#endif

'''
    code = code[:position] + extra + code[position:]
    anchor = 'template <> void gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>'
    position = code.index(anchor)
    extra = GUARD + '''template <> void gemm<block_mxfp4, 16, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, n);
    for (int r = 0; r < nr; ++r) {
        gemv<block_mxfp4, 16, 4, GGML_TYPE_Q8_0>(n, s + r * bs, bs, vx, (const char *) vy + r * stride, 1, nc);
    }
}
#endif

'''
    code = code[:position] + extra + code[position:]
    anchor = '    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + '\n' + GUARD + '    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 16, 4, GGML_TYPE_Q8_0> mxfp4_4x16_q8_0;\n#endif')
    anchor = '    } else if (cur->type == GGML_TYPE_MXFP4) {\n'
    assert code.count(anchor) == 1
    extra = GUARD + '''        static const bool avx_packed = [] {
            const char * env = std::getenv("GGML_CPU_MXFP4_PACKED");
            return env != nullptr && std::strcmp(env, "1") == 0;
        }();
        if (avx_packed && ggml_cpu_has_avx() && ggml_n_dims(cur) == 3 &&
            cur->ne[0] > 0 && cur->ne[0] <= INT32_MAX && cur->ne[0] % QK_MXFP4 == 0 &&
            cur->ne[1] > 0 && cur->ne[1] % 4 == 0 && cur->ne[2] > 0 && cur->ne[1] <= INT32_MAX / cur->ne[2]) {
            return &mxfp4_4x16_q8_0;
        }
#endif
'''
    code = code.replace(anchor, anchor + extra)
    repack.write_text(fix_mapping(code))
    patches = []
    for path in (rel / 'arch/x86/quants.c', rel / 'quants.h', rel / 'repack.cpp'):
        patches += difflib.unified_diff((BASE / path).read_text().splitlines(True), (source / path).read_text().splitlines(True), fromfile='a/' + str(path), tofile='b/' + str(path))
    (ROOT / 'packed.patch').write_text(''.join(patches))
    commands = json.loads((LAB / 'review-mxfp4-20261023/build-commands.json').read_text())
    commands = [[x.replace(str(LAB / 'review-mxfp4-20261023'), str(ROOT)) for x in row] for row in commands]
    (ROOT / 'build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    for label, command in zip(('configure', 'build'), commands):
        print(label + ' packed runtime', flush=True)
        with (ROOT / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Packed runtime complete', flush=True)


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
