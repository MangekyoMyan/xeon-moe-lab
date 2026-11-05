#!/usr/bin/env python3
"""Extend the standalone comparison with equal-size packed matrices."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'review-glm-q8-columns4-20261020'


def main():
    code = (ROOT / 'probe.cpp').read_text()
    code = code.replace('dot_fn dot; int rows; };', 'dot_fn dot; int rows; bool packed; };')
    code = code.replace('vars.push_back({name, handle, dot, rows});', 'vars.push_back({name, handle, dot, rows, name.find("packed") != std::string::npos});')
    anchor = 'static void check('
    pack = '''static void pack_four(unsigned char * dst, const char * src, size_t stride, int k) {
    for (int ib = 0; ib < k / 32; ++ib) {
        for (int r = 0; r < 4; ++r) {
            const auto * row = (const block_mxfp4 *) (src + r * stride);
            dst[size_t(ib) * 68 + r] = row[ib].e;
            std::memcpy(dst + size_t(ib) * 68 + 4 + r * 16, row[ib].qs, 16);
        }
    }
}

'''
    code = code.replace(anchor, pack + anchor)
    anchor = '            v.dot(k, outputs.data(), 0, grouped.data(), stride, y, 0, v.rows);'
    replacement = '''            std::vector<unsigned char> packed(stride * 4);
            if (v.packed) pack_four(packed.data(), (const char *) grouped.data(), stride, k);
            v.dot(k, outputs.data(), 0, v.packed ? (const void *) packed.data() : (const void *) grouped.data(), stride, y, 0, v.rows);'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    code = code.replace('        if (v.rows == 1) continue;', '        if (v.rows == 1 || v.packed) continue;')
    anchor = '    std::printf("{\\\"guarded_group_inputs\\\"'
    pos = code.index(anchor)
    guard = '''    uint64_t packed_guards = 0;
    for (const auto & v : vars) {
        if (!v.packed) continue;
        for (int nb : {0, 1, 2, 3, 5, 71, 72, 90, 160}) {
            const size_t stride = nb * sizeof(block_mxfp4), span = 4 * stride;
            const size_t xp = (span + page - 1) / page * page;
            char * xm = (char *) mmap(nullptr, xp + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            assert(xm != MAP_FAILED && !mprotect(xm + xp, page, PROT_NONE));
            auto * packed = (unsigned char *) (xm + xp - span);
            std::vector<block_mxfp4> raw(4 * nb);
            std::vector<block_q8_0> rhs(nb);
            for (uint32_t seed = 1; seed <= 32; ++seed) {
                for (int r = 0; r < 4; ++r) fill(raw.data() + r * nb, rhs.data(), nb, seed + r * 97);
                pack_four(packed, (const char *) raw.data(), stride, nb * 32);
                float actual[4];
                v.dot(nb * 32, actual, 0, packed, stride, rhs.data(), 0, 4);
                for (int r = 0; r < 4; ++r) {
                    float expected;
                    ggml_vec_dot_mxfp4_q8_0(nb * 32, &expected, 0, raw.data() + r * nb, 0, rhs.data(), 0, 1);
                    assert(!std::memcmp(&expected, &actual[r], sizeof(float)));
                }
                ++packed_guards;
            }
            munmap(xm, xp + page);
        }
    }
    std::printf("{\\\"guarded_packed_inputs\\\":%llu}\\n", (unsigned long long) packed_guards);
'''
    code = code[:pos] + guard + code[pos:]
    code = code.replace('const variant & v, const char * x, const void * y,', 'const variant & v, const char * x, const char * packed, const void * y,')
    code = code.replace('const int r0 = int(int64_t(m) * tid / threads), r1 = int(int64_t(m) * (tid + 1) / threads);', 'const int r0 = int(int64_t(m / 4) * tid / threads) * 4, r1 = int(int64_t(m / 4) * (tid + 1) / threads) * 4;')
    code = code.replace('v.dot(k, values, 0, x + size_t(r) * stride, stride, y, 0, v.rows);', 'v.dot(k, values, 0, (v.packed ? packed : x) + size_t(r) * stride, stride, y, 0, v.rows);')
    code = code.replace('m < 1 || m > 262144', 'm < 4 || m % 4 || m > 262144')
    anchor = '        std::vector<int> iterations;'
    replacement = '''        auto * packed = (char *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(packed != MAP_FAILED && !madvise(packed, bytes, MADV_NOHUGEPAGE));
        const double packing_start = omp_get_wtime();
        #pragma omp parallel num_threads(threads)
        {
            pin(omp_get_thread_num());
            #pragma omp for schedule(static)
            for (int r = 0; r < m; r += 4) pack_four((unsigned char *) packed + size_t(r) * stride, x + size_t(r) * stride, stride, k);
        }
        std::printf("{\\\"packed_bytes\\\":%zu,\\\"original_bytes\\\":%zu,\\\"packing_s\\\":%.9f}\\n", bytes, bytes, omp_get_wtime() - packing_start);
        std::vector<int> iterations;'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    code = code.replace('measure(v, x, y.data()', 'measure(v, x, packed, y.data()')
    code = code.replace('        munmap(x, bytes);', '        munmap(x, bytes);\n        munmap(packed, bytes);')
    source = ROOT / 'packed-probe.cpp'
    source.write_text(code)
    command = ['c++', '-O3', '-march=ivybridge', '-std=c++17', '-fopenmp', '-I' + str(BASE / 'source/ggml/include'), '-I' + str(BASE / 'source/ggml/src'), str(source), '-L' + str(BASE / 'build/bin'), '-Wl,-rpath,' + str(BASE / 'build/bin'), '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(ROOT / 'packed-probe')]
    subprocess.run(command, check=True)
    (ROOT / 'packed-probe-build.json').write_text(json.dumps(command, indent=2) + '\n')


if __name__ == '__main__':
    main()
