#!/usr/bin/env python3
"""Build isolated, bit-preserving AVX MXFP4 kernel candidates."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-glm-q8-columns4-20261020'
SOURCE = BASE / 'source/ggml/src/ggml-cpu/arch/x86/quants.c'
original = SOURCE.read_text()
start = original.index('void ggml_vec_dot_mxfp4_q8_0(')
avx = original.index('#elif defined __AVX__', start)
end = original.index('\n#endif', avx)
body = original[avx:end]


def narrow(code):
    old = '        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);'
    new = '''        // Four products fit in int16: 4 * 12 * 128 = 6144.
        const __m128i p16_1 = _mm_add_epi16(mul_add_epi8_sse(q4b_1_0, q8b_1_0), mul_add_epi8_sse(q4b_1_1, q8b_1_1));
        const __m128i p16_2 = _mm_add_epi16(mul_add_epi8_sse(q4b_2_0, q8b_2_0), mul_add_epi8_sse(q4b_2_1, q8b_2_1));
        const __m128i p_1 = _mm_madd_epi16(p16_1, _mm_set1_epi16(1));
        const __m128i p_2 = _mm_madd_epi16(p16_2, _mm_set1_epi16(1));
        const __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));'''
    assert code.count(old) == 1
    return code.replace(old, new)


def delta(code, mode):
    old = '        const __m256 deltas = quad_mx_delta_float(x[ib].e, y[ib].d, x[ib + 1].e, y[ib + 1].d);'
    if mode == 'typed':
        new = '''        const __m256 deltas = _mm256_set_m128(
            _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d)),
            _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e) * GGML_CPU_FP16_TO_FP32(y[ib].d)));'''
    elif mode == 'f16c':
        new = '''        const __m256 deltas = _mm256_set_m128(
            _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e) * _cvtsh_ss(y[ib + 1].d)),
            _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e) * _cvtsh_ss(y[ib].d)));'''
    else:
        new = '''        const uint32_t halves = (uint32_t) y[ib].d | ((uint32_t) y[ib + 1].d << 16);
        const __m128 ys = _mm_cvtph_ps(_mm_cvtsi32_si128((int32_t) halves));
        const __m128 ds = _mm_mul_ps(ys, _mm_setr_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e), GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e), 0, 0));
        const __m256 deltas = _mm256_set_m128(_mm_shuffle_ps(ds, ds, 0x55), _mm_shuffle_ps(ds, ds, 0));'''
    assert code.count(old) == 1
    return code.replace(old, new)


def sse(code):
    code = code.replace('    __m256 accum = _mm256_setzero_ps();', '    __m128 accum0 = _mm_setzero_ps();\n    __m128 accum1 = _mm_setzero_ps();')
    start = code.index('        const __m256 p = _mm256_cvtepi32_ps(')
    stop = code.index('\n    }', start)
    code = code[:start] + '''        const __m128 d0 = _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e) * GGML_CPU_FP16_TO_FP32(y[ib].d));
        const __m128 d1 = _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d));
        accum0 = _mm_add_ps(_mm_mul_ps(d0, _mm_cvtepi32_ps(p_1)), accum0);
        accum1 = _mm_add_ps(_mm_mul_ps(d1, _mm_cvtepi32_ps(p_2)), accum1);''' + code[stop:]
    return code.replace('    sumf = hsum_float_8(accum);', '    sumf = hsum_float_8(_mm256_set_m128(accum1, accum0));')


def prefetch(code, distance):
    anchor = '    for (; ib + 1 < nb; ib += 2) {\n'
    return code.replace(anchor, anchor + f'        if (ib + {distance} < nb) {{\n            _mm_prefetch((const char *) &x[ib + {distance}], _MM_HINT_T0);\n        }}\n')


def main():
    variants = [('baseline', body), ('narrow', narrow(body)), ('typed', delta(body, 'typed')),
                ('narrow-typed', delta(narrow(body), 'typed')),
                ('narrow-f16c', delta(narrow(body), 'f16c')),
                ('narrow-f16cpair', delta(narrow(body), 'pair')),
                ('narrow-sse', sse(narrow(body)))]
    variants += [(f'narrow-pf{d}', prefetch(narrow(body), d)) for d in (8, 16, 32)]
    entry = next(x for x in json.loads((BASE / 'build/compile_commands.json').read_text()) if x['file'] == str(SOURCE))
    flags = shlex.split(entry['command'])
    flags = flags[:flags.index('-o')]
    flags = ['-march=ivybridge' if f == '-march=native' else f for f in flags]
    result = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(original.encode()).hexdigest(), 'variants': []}
    for name, code in variants:
        text = original[:avx] + code + original[end:]
        path = ROOT / (name + '.c')
        path.write_text(text)
        (ROOT / (name + '.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/ggml/src/ggml-cpu/arch/x86/quants.c', tofile='b/ggml/src/ggml-cpu/arch/x86/quants.c')))
        command = flags + ['-iquote', str(SOURCE.parent), '-shared', '-Wl,-Bsymbolic', str(path), '-L' + str(BASE / 'build/bin'), '-Wl,-rpath,' + str(BASE / 'build/bin'), '-lggml-cpu', '-lggml-base', '-lm', '-o', str(ROOT / (name + '.so'))]
        with (ROOT / (name + '.compile.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        asm = subprocess.check_output(['objdump', '-d', '--disassemble=ggml_vec_dot_mxfp4_q8_0', str(ROOT / (name + '.so'))], text=True)
        (ROOT / (name + '.asm')).write_text(asm)
        result['variants'].append({'name': name, 'command': command, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'binary_sha256': hashlib.sha256((ROOT / (name + '.so')).read_bytes()).hexdigest()})
        (ROOT / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
        print('compiled ' + name, flush=True)


if __name__ == '__main__':
    main()
