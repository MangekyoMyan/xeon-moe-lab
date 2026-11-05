#!/usr/bin/env python3
"""Use the existing TinyBLAS accumulation order when testing dense reuse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'review-q8-reuse-20261025/source'
REL = Path('ggml/src/ggml-cpu')

TEMPLATE = r'''
void ggml_vec_dot_q8_0_q8_0_sgemm_colsCOLS(int n, float * GGML_RESTRICT sums, const void * vx, const void * const vy[COLS]) {
    assert(n >= 0 && n % QK8_0 == 0);
    const block_q8_0 * x = vx;
    const block_q8_0 * y[COLS];
    __m256 acc[COLS];
    for (int c = 0; c < COLS; ++c) {
        y[c] = vy[c];
        acc[c] = _mm256_setzero_ps();
    }
    const __m128i ones = _mm_set1_epi16(1);
    for (int block = 0; block < n / QK8_0; ++block) {
        const __m128i x0 = _mm_loadu_si128((const __m128i *) x[block].qs);
        const __m128i x1 = _mm_loadu_si128((const __m128i *) (x[block].qs + 16));
        const __m128i a0 = _mm_sign_epi8(x0, x0);
        const __m128i a1 = _mm_sign_epi8(x1, x1);
        const float xd = GGML_CPU_FP16_TO_FP32(x[block].d);
        for (int c = 0; c < COLS; ++c) {
            const __m128i y0 = _mm_loadu_si128((const __m128i *) y[c][block].qs);
            const __m128i y1 = _mm_loadu_si128((const __m128i *) (y[c][block].qs + 16));
            const __m128i p0 = _mm_madd_epi16(ones, _mm_maddubs_epi16(a0, _mm_sign_epi8(y0, x0)));
            const __m128i p1 = _mm_madd_epi16(ones, _mm_maddubs_epi16(a1, _mm_sign_epi8(y1, x1)));
            const __m256 product = _mm256_cvtepi32_ps(MM256_SET_M128I(p1, p0));
            const __m256 scale = _mm256_set1_ps(xd * GGML_CPU_FP16_TO_FP32(y[c][block].d));
            acc[c] = _mm256_add_ps(_mm256_mul_ps(scale, product), acc[c]);
        }
    }
    for (int c = 0; c < COLS; ++c) {
        // Match llamafile::hsum exactly, including its pairwise addition order.
        __m128 sum = _mm_add_ps(_mm256_extractf128_ps(acc[c], 1), _mm256_castps256_ps128(acc[c]));
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
        sums[c] = _mm_cvtss_f32(sum);
    }
}
'''

def main():
    qfile = SOURCE / REL / 'arch/x86/quants.c'
    text = qfile.read_text()
    assert 'sgemm_cols4' not in text
    text += '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'
    for cols in [1, 2, 4]:
        text += TEMPLATE.replace('COLS', str(cols))
    text += '#endif\n'
    qfile.write_text(text)
    header = SOURCE / REL / 'quants.h'
    text = header.read_text()
    anchor = next(line for line in text.splitlines() if line.startswith('void ggml_vec_dot_q8_0_q8_0_cols4('))
    declarations = [TEMPLATE.replace('COLS', str(cols)).splitlines()[1].replace(' {', ';') for cols in [1, 2, 4]]
    text = text.replace(anchor, anchor + '\n' + '\n'.join(declarations))
    header.write_text(text)
    cpu = SOURCE / REL / 'ggml-cpu.c'
    text = cpu.read_text()
    anchor = 'if (ggml_cpu_q8_dense_cols && !params->use_ref && type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1) {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'if (ggml_cpu_q8_dense_cols && !params->use_ref && type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1 && ne11 > 1) {')
    start = text.index('if (ggml_cpu_q8_dense_cols && !params->use_ref')
    end = text.index('\n#endif', start)
    block = text[start:end]
    block = block.replace('int columns = 0;', 'int columns = 1;')
    block = block.replace('ggml_vec_dot_q8_0_q8_0_cols4(', 'ggml_vec_dot_q8_0_q8_0_sgemm_cols4(')
    block = block.replace('else ggml_vec_dot_q8_0_q8_0_cols2(ne00, sums, src0_row + row * nb01, inputs);',
                          'else if (columns == 2) ggml_vec_dot_q8_0_q8_0_sgemm_cols2(ne00, sums, src0_row + row * nb01, inputs);\n                            else ggml_vec_dot_q8_0_q8_0_sgemm_cols1(ne00, sums, src0_row + row * nb01, inputs);')
    text = text[:start] + block + text[end:]
    # The opt-in columns must bypass TinyBLAS; otherwise the experiment is a no-op.
    start = text.index('void ggml_compute_forward_mul_mat(\n')
    end = text.index('// ggml_compute_forward_mul_mat_id', start)
    body = text[start:end]
    anchor = '    const bool src1_cont = ggml_is_contiguous(src1);'
    assert body.count(anchor) == 1
    body = body.replace(anchor, anchor + '''

    const bool custom_dense_columns =
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
        ggml_cpu_q8_dense_cols && !params->use_ref && src0->type == GGML_TYPE_Q8_0 && ne11 > 1;
#else
        false;
#endif''')
    body = body.replace('    if (src1_cont) {', '    if (src1_cont && !custom_dense_columns) {')
    anchor = '#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {'
    assert body.count(anchor) == 1
    body = body.replace(anchor, '#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type && !custom_dense_columns) {')
    text = text[:start] + body + text[end:]
    cpu.write_text(text)
    print('Dense column experiments now preserve TinyBLAS arithmetic and bypass its original dispatch')

if __name__ == '__main__':
    main()
