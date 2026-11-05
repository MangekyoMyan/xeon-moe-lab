#!/usr/bin/env python3
"""Generate exact-order Q8 AVX1 experiments without changing the accepted runtime."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-glm-q8-columns4-20261020'
PATH = BASE / 'source/ggml/src/ggml-cpu/arch/x86/quants.c'

HELPERS = r'''
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
static inline __m128i ivb_mul_abs(const __m128i x, const __m128i y) {
    return _mm_maddubs_epi16(_mm_abs_epi8(x), _mm_sign_epi8(y, x));
}

static inline __m256 ivb_quad_abs(const __m128i x0, const __m128i x1, const __m128i x2, const __m128i x3,
                                const __m128i y0, const __m128i y1, const __m128i y2, const __m128i y3) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i p0 = _mm_add_epi32(_mm_madd_epi16(ivb_mul_abs(x0, y0), ones), _mm_madd_epi16(ivb_mul_abs(x1, y1), ones));
    const __m128i p1 = _mm_add_epi32(_mm_madd_epi16(ivb_mul_abs(x2, y2), ones), _mm_madd_epi16(ivb_mul_abs(x3, y3), ones));
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p1, p0));
}

static inline __m256 ivb_delta_typed(ggml_half x0, ggml_half y0, ggml_half x1, ggml_half y1) {
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                          _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 ivb_delta_f16c(ggml_half x0, ggml_half y0, ggml_half x1, ggml_half y1) {
    const __m128 fp = _mm_cvtph_ps(_mm_set_epi16(0, 0, 0, 0, y1, x1, y0, x0));
    const __m128 px = _mm_shuffle_ps(fp, fp, _MM_SHUFFLE(2, 2, 0, 0));
    const __m128 py = _mm_shuffle_ps(fp, fp, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128 products = _mm_mul_ps(px, py);
    return _mm256_set_m128(_mm_shuffle_ps(products, products, _MM_SHUFFLE(2, 2, 2, 2)),
                          _mm_shuffle_ps(products, products, _MM_SHUFFLE(0, 0, 0, 0)));
}

static inline __m256 ivb_delta_f16c_scalar(ggml_half x0, ggml_half y0, ggml_half x1, ggml_half y1) {
    return _mm256_set_m128(_mm_set1_ps(_cvtsh_ss(x1) * _cvtsh_ss(y1)),
                          _mm_set1_ps(_cvtsh_ss(x0) * _cvtsh_ss(y0)));
}
#endif
'''


def make_variant(original, name, abs_op=False, scale='table', unroll=1, distance=8, hint='T0'):
    code = original.replace('ggml_vec_dot_q8_0_q8_0(', name + '(')
    start = code.index('#elif defined(__AVX__)')
    end = code.index('#endif', start)
    quad = 'ivb_quad_abs' if abs_op else 'mul_sum_i8_quad_float'
    delta = {'table': 'quad_fp16_delta_float', 'typed': 'ivb_delta_typed',
             'f16c': 'ivb_delta_f16c', 'f16c_scalar': 'ivb_delta_f16c_scalar'}[scale]
    body = ['#elif defined(__AVX__)', '    __m256 accum = _mm256_setzero_ps();']
    def pair(index, suffix):
        lines = []
        for var in ('x', 'y'):
            for block in (0, 1):
                for half in (0, 1):
                    lines.append(f'        const __m128i {var}{suffix}_{block}_{half} = _mm_loadu_si128((const __m128i *) {var}[{index} + {block}].qs + {half});')
        args = ', '.join(f'{var}{suffix}_{block}_{half}' for var in ('x', 'y') for block in (0, 1) for half in (0, 1))
        lines += [f'        const __m256 p{suffix} = {quad}({args});',
                  f'        const __m256 d{suffix} = {delta}(x[{index}].d, y[{index}].d, x[{index} + 1].d, y[{index} + 1].d);']
        return lines
    body.append(f'    for (; ib + {2*unroll-1} < nb; ib += {2*unroll}) {{')
    if distance:
        body.append(f'        if (ib + {distance} < nb) _mm_prefetch((const char *) &x[ib + {distance}], _MM_HINT_{hint});')
    for j in range(unroll):
        body += pair('ib' if j == 0 else f'ib + {2*j}', str(j))
    for j in range(unroll):
        body.append(f'        accum = _mm256_add_ps(_mm256_mul_ps(d{j}, p{j}), accum);')
    body.append('    }')
    if unroll > 1:
        body.append('    for (; ib + 1 < nb; ib += 2) {')
        body += pair('ib', 'tail')
        body += ['        accum = _mm256_add_ps(_mm256_mul_ps(dtail, ptail), accum);', '    }']
    body += ['    sumf = hsum_float_8(accum);', '']
    return code[:start] + '\n'.join(body) + code[end:]


def main():
    original = PATH.read_text()
    start = original.index('void ggml_vec_dot_q8_0_q8_0(')
    end = original.index('\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)', start)
    fn = original[start:end]
    variants = [
        ('abs', dict(abs_op=True)), ('typed', dict(scale='typed')), ('f16c', dict(scale='f16c')),
        ('f16c_scalar', dict(scale='f16c_scalar')), ('abs_f16c', dict(abs_op=True, scale='f16c')),
        ('u4', dict(unroll=2)), ('u8', dict(unroll=4)),
        ('abs_f16c_u4', dict(abs_op=True, scale='f16c', unroll=2)),
        ('pf0', dict(distance=0)), ('pf16', dict(distance=16)), ('pf32', dict(distance=32)), ('pf64', dict(distance=64)),
        ('pf32_t1', dict(distance=32, hint='T1')), ('pf32_nta', dict(distance=32, hint='NTA')),
    ]
    code = original + '\n' + HELPERS + '\n'
    for name, options in variants:
        candidate = make_variant(fn, 'ivb_q8_' + name, **options)
        (ROOT / (name + '-function.c')).write_text(candidate)
        code += '\n' + candidate
    (ROOT / 'kernels.c').write_text(code)
    entry = next(x for x in json.loads((BASE / 'build/compile_commands.json').read_text()) if x['file'] == str(PATH))
    flags = shlex.split(entry['command']); flags = flags[:flags.index('-o')]
    lib = BASE / 'build/bin'
    command = flags + ['-iquote', str(PATH.parent), '-shared', '-Wl,-Bsymbolic', str(ROOT / 'kernels.c'),
                       '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lggml-cpu', '-lggml-base', '-lm', '-o', str(ROOT / 'kernels.so')]
    with (ROOT / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    symbols = {'baseline':'ggml_vec_dot_q8_0_q8_0', **{name:'ivb_q8_'+name for name, _ in variants}}
    (ROOT / 'kernel-variants.json').write_text(json.dumps(symbols, indent=2)+'\n')
    (ROOT / 'kernel-build.json').write_text(json.dumps({'command':command,'baseline_source_sha256':hashlib.sha256(original.encode()).hexdigest()},indent=2)+'\n')
    print('Built',len(symbols),'variants',flush=True)


if __name__ == '__main__':
    main()
