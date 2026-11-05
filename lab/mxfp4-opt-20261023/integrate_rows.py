#!/usr/bin/env python3
"""Add opt-in row sharing to the isolated AVX MoE runtime."""
import difflib
from pathlib import Path
from prepare_rows import kernel

LAB = Path(__file__).resolve().parent.parent
ROOT = LAB / 'review-mxfp4-20261023'
BASE = LAB / 'review-glm-q8-columns4-20261020/source'


def main():
    source = ROOT / 'source'
    rel = Path('ggml/src/ggml-cpu')
    qfile = source / rel / 'arch/x86/quants.c'
    code = qfile.read_text()
    functions = []
    declarations = []
    for rows in (2, 4):
        fn = kernel(rows, True)
        before = 'void ggml_vec_dot_mxfp4_q8_0_grouped(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc)'
        after = f'void ggml_vec_dot_mxfp4_q8_0_rows{rows}(int n, float * s, const void * vx, size_t bx, const void * vy)'
        fn = fn.replace(before, after)
        fn = fn.replace('    if (nrc == 1) { ggml_vec_dot_mxfp4_q8_0(n, s, bs, vx, bx, vy, by, nrc); return; }\n', '')
        fn = fn.replace(f'assert(nrc == {rows} && n >= 0 && n % 32 == 0);', 'assert(n >= 0 && n % 32 == 0);')
        fn = fn.replace('    const __m128i values =', '    // Bias FP4 values by 12; subtract the shared input contribution in int16.\n    const __m128i values =')
        functions.append(fn)
        declarations.append(after + ';')
    anchor = '\nvoid ggml_vec_dot_nvfp4_q8_0('
    assert code.count(anchor) == 1 and 'ggml_vec_dot_mxfp4_q8_0_rows4' not in code
    code = code.replace(anchor, '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n' + '\n'.join(functions) + '\n#endif\n' + anchor)
    qfile.write_text(code)
    header = source / rel / 'quants.h'
    code = header.read_text()
    declaration = next(x for x in code.splitlines() if x.startswith('void ggml_vec_dot_q8_0_q8_0_rows4('))
    code = code.replace(declaration, declaration + '\n' + '\n'.join(declarations))
    header.write_text(code)
    cpu = source / rel / 'ggml-cpu.c'
    code = cpu.read_text()
    code = code.replace('float ggml_table_f32_e8m0_half[1 << 8];', 'float ggml_table_f32_e8m0_half[1 << 8];\n\nstatic int ggml_cpu_mxfp4_rows = 0;')
    anchor = '''                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }'''
    replacement = '''                int64_t ir0 = iir0;
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
                if (!use_ref && type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32 && ids->ne[1] == 1 && ggml_cpu_mxfp4_rows != 0) {
                    const int rows = ggml_cpu_mxfp4_rows;
                    for (; ir0 + rows <= iir0 + blck_0 && ir0 + rows <= ir0_end; ir0 += rows) {
                        if (rows == 4) {
                            ggml_vec_dot_mxfp4_q8_0_rows4(ne00, &tmp[ir0 - iir0], src0_cur + ir0 * nb01, nb01, src1_col);
                        } else {
                            ggml_vec_dot_mxfp4_q8_0_rows2(ne00, &tmp[ir0 - iir0], src0_cur + ir0 * nb01, nb01, src1_col);
                        }
                    }
                }
#endif
                for (; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    anchor = '            env = getenv("GGML_CPU_MXFP4_PAIR");'
    assert code.count(anchor) == 1
    code = code.replace(anchor, '            env = getenv("GGML_CPU_MXFP4_ROWS");\n            ggml_cpu_mxfp4_rows = env != NULL && strcmp(env, "4") == 0 ? 4 : (env != NULL && strcmp(env, "2") == 0 ? 2 : 0);\n' + anchor)
    cpu.write_text(code)
    patch = []
    for path in (rel / 'arch/x86/quants.c', rel / 'quants.h', rel / 'ggml-cpu.c'):
        patch += difflib.unified_diff((BASE / path).read_text().splitlines(True), (source / path).read_text().splitlines(True), fromfile='a/' + str(path), tofile='b/' + str(path))
    (ROOT / 'candidate.patch').write_text(''.join(patch))


if __name__ == '__main__':
    main()
