#!/usr/bin/env python3
"""Give ordinary Q8 decode matrices the same optional row-placement contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPU = ROOT.parent / 'review-q8-reuse-20261025/source/ggml/src/ggml-cpu/ggml-cpu.c'

def main():
    text = CPU.read_text()
    anchor = 'static int ggml_cpu_q8_dense_cols = 0;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\nstatic int ggml_cpu_q8_dense_rows = 0;')
    anchor = '''    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.'''
    assert text.count(anchor) == 1
    replacement = '''    const int64_t nr1 = ne1 * ne2 * ne3;

    if (ggml_cpu_q8_dense_rows && !params->use_ref && src0->type == GGML_TYPE_Q8_0 && nr1 == 1) {
        int rank = ith;
        if (ggml_cpu_q8_dense_rows == 2 && nth == 24) {
            // Match the private two-socket worker permutation used by MoE.
            rank = (ith % 6) + (ith >= 12 ? 6 : 0) + (ith % 12 >= 6 ? 12 : 0);
        }
        const int64_t begin = nr0 * rank / nth;
        const int64_t end = nr0 * (rank + 1) / nth;
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, 1, begin, end, 0, 1);
        return;
    }

    // Now select a reasonable chunk size.'''
    text = text.replace(anchor, replacement)
    anchor = '            const char * dense_cols = getenv("GGML_CPU_Q8_DENSE_COLS");'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
            const char * dense_rows = getenv("GGML_CPU_Q8_DENSE_ROWS");
            ggml_cpu_q8_dense_rows = dense_rows ? atoi(dense_rows) : 0;
            if (ggml_cpu_q8_dense_rows < 0 || ggml_cpu_q8_dense_rows > 2) ggml_cpu_q8_dense_rows = 0;''')
    CPU.write_text(text)
    print('Added ordinary Q8 static row schedules; rebuild required')

if __name__ == '__main__':
    main()
