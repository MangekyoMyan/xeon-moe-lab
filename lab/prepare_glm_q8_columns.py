#!/usr/bin/env python3
"""Private AVX Q8 MoE prefill experiment: share a weight row across four inputs."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

LAB = Path(__file__).resolve().parent
ROOT = LAB / "review-glm-q8-columns4-20261020"

KERNEL = """
void ggml_vec_dot_q8_0_q8_0_cols4(int n, float * GGML_RESTRICT s, const void * vx, const void * const vy[4]) {
    assert(n >= 0 && n % QK8_0 == 0);
    const int nb = n / QK8_0;
    const block_q8_0 * x = vx;
    const block_q8_0 * y[4] = { vy[0], vy[1], vy[2], vy[3] };
    __m256 acc[4];
    for (int c = 0; c < 4; ++c) {
        acc[c] = _mm256_setzero_ps();
    }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i x00 = _mm_loadu_si128((const __m128i *) x[ib].qs);
        const __m128i x01 = _mm_loadu_si128((const __m128i *) (x[ib].qs + 16));
        const __m128i x10 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
        const __m128i x11 = _mm_loadu_si128((const __m128i *) (x[ib + 1].qs + 16));
        const float xd0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const float xd1 = GGML_CPU_FP16_TO_FP32(x[ib + 1].d);
        for (int c = 0; c < 4; ++c) {
            const __m128i y00 = _mm_loadu_si128((const __m128i *) y[c][ib].qs);
            const __m128i y01 = _mm_loadu_si128((const __m128i *) (y[c][ib].qs + 16));
            const __m128i y10 = _mm_loadu_si128((const __m128i *) y[c][ib + 1].qs);
            const __m128i y11 = _mm_loadu_si128((const __m128i *) (y[c][ib + 1].qs + 16));
            const __m256 p = mul_sum_i8_quad_float(x00, x01, x10, x11, y00, y01, y10, y11);
            const __m256 d = _mm256_set_m128(
                _mm_set1_ps(xd1 * GGML_CPU_FP16_TO_FP32(y[c][ib + 1].d)),
                _mm_set1_ps(xd0 * GGML_CPU_FP16_TO_FP32(y[c][ib].d)));
            acc[c] = _mm256_add_ps(_mm256_mul_ps(d, p), acc[c]);
        }
    }
    for (int c = 0; c < 4; ++c) {
        float sum = hsum_float_8(acc[c]);
        if (ib < nb) {
            int sumi = 0;
            for (int j = 0; j < QK8_0; ++j) {
                sumi += x[ib].qs[j] * y[c][ib].qs[j];
            }
            sum += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[c][ib].d));
        }
        s[c] = sum;
    }
}
"""

DISPATCH = """            int64_t ir1 = iir1;
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
            if (!use_ref && type == GGML_TYPE_Q8_0 && ids->ne[1] > 1) {
                for (; ir1 + 3 < iir1 + blck_1 && ir1 + 3 < ir1_end; ir1 += 4) {
                    const void * columns[4];
                    float * outputs[4];
                    for (int c = 0; c < 4; ++c) {
                        const struct mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, ir1 + c);
                        const int64_t i11 = mapping.i1 % ne11;
                        const int64_t i12 = mapping.i2;
                        columns[c] = (const char *) wdata +
                            (src1_cont || src1->type != vec_dot_type
                                ? (i11 + i12 * ne11) * row_size
                                : i11 * nb11 + i12 * nb12);
                        outputs[c] = (float *) ((char *) dst->data + mapping.i1 * nb1 + i12 * nb2);
                    }
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        float sums[4];
                        ggml_vec_dot_q8_0_q8_0_cols4(ne00, sums, src0_cur + ir0 * nb01, columns);
                        for (int c = 0; c < 4; ++c) {
                            outputs[c][ir0] = sums[c];
                        }
                    }
                }
            }
#else
            GGML_UNUSED(use_ref);
#endif
            for (; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {"""


def main():
    base = LAB / "llama-private-prefetch"
    ROOT.mkdir(exist_ok=False)
    shutil.copytree(base, ROOT / "source")
    source = ROOT / "source"
    relative = Path("ggml/src/ggml-cpu")
    qfile = source / relative / "arch/x86/quants.c"
    text = qfile.read_text()
    start = text.index("void ggml_vec_dot_q8_0_q8_0_rows4(")
    end = text.index("\n#endif", start)
    qfile.write_text(text[:end] + "\n" + KERNEL + text[end:])
    header = source / relative / "quants.h"
    text = header.read_text()
    declaration = next(line for line in text.splitlines()
                       if line.startswith("void ggml_vec_dot_q8_0_q8_0_rows4("))
    text = text.replace(declaration, declaration + "\n" + KERNEL.splitlines()[1].replace(" {", ";"))
    header.write_text(text)
    cpu = source / relative / "ggml-cpu.c"
    text = cpu.read_text()
    anchor = "static void ggml_compute_forward_mul_mat_id_one_chunk(\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + "    const bool use_ref,\n")
    for call in ("dst, src0, src1, ids, cur_a,", "pair, pair_weight, src1, ids, cur_a,"):
        assert text.count(call) == 1
        text = text.replace(call, "params->use_ref, " + call)
    start = text.index(anchor)
    end = text.index("\nstatic void * incr_ptr_aligned", start)
    fn = text[start:end]
    anchor = "            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {"
    assert fn.count(anchor) == 1
    fn = fn.replace(anchor, DISPATCH)
    cpu.write_text(text[:start] + fn + text[end:])
    patch = []
    hashes = {}
    for rel in [relative / "arch/x86/quants.c", relative / "quants.h", relative / "ggml-cpu.c"]:
        before, after = (base / rel).read_bytes(), (source / rel).read_bytes()
        hashes[str(rel)] = {"before": hashlib.sha256(before).hexdigest(),
                            "after": hashlib.sha256(after).hexdigest()}
        patch.extend(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                         fromfile="a/" + str(rel), tofile="b/" + str(rel)))
    (ROOT / "candidate.patch").write_text("".join(patch))
    configure = ["cmake", "-S", str(source), "-B", str(ROOT / "build"),
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DGGML_NATIVE=ON", "-DGGML_OPENMP=ON", "-DGGML_BLAS=OFF",
                 "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_OPENCL=OFF",
                 "-DGGML_METAL=OFF", "-DGGML_SYCL=OFF", "-DLLAMA_BUILD_UI=OFF",
                 "-DLLAMA_BUILD_MTMD=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TESTS=ON"]
    build = ["cmake", "--build", str(ROOT / "build"), "--target", "llama-bench", "llama-completion",
             "test-backend-ops", "test-quantize-fns", "-j", "8"]
    (ROOT / "prepare.json").write_text(json.dumps(
        {"hashes": hashes, "commands": [configure, build]}, indent=2) + "\n")
    for label, command in (("configure", configure), ("build", build)):
        print(label, ROOT, flush=True)
        with (ROOT / (label + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
