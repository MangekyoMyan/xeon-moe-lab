#!/usr/bin/env python3
"""Explore AVX1 Q8 TinyBLAS tile sizes and aligned prefill NUMA work."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-q8-reuse-20261025'
DEST = LAB / 'review-q8-tiles-20261025'

def main():
    DEST.mkdir(exist_ok=False)
    shutil.copytree(BASE / 'source', DEST / 'source')
    cpu_dir = DEST / 'source/ggml/src/ggml-cpu'
    sgemm = cpu_dir / 'llamafile/sgemm.cpp'
    text = sgemm.read_text()
    anchor = 'inline float unhalf(ggml_fp16_t d) {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'static int q8_tile_code = 0;\n\n' + anchor)
    start = text.index('class tinyBLAS_Q0_AVX {')
    anchor = '        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {'
    location = text.index(anchor, start)
    choices = []
    for code, rows, cols in [(12, 1, 2), (14, 1, 4), (22, 2, 2), (23, 2, 3), (24, 2, 4), (32, 3, 2), (41, 4, 1), (42, 4, 2)]:
        choices.append(f'''        if (std::is_same<TA, block_q8_0>::value && std::is_same<TB, block_q8_0>::value &&
                q8_tile_code == {code} && m - m0 >= {rows} && n - n0 >= {cols}) {{
            mc = {rows}; nc = {cols};
            gemm<{rows}, {cols}>(m0, m, n0, n);
        }} else''')
    override = '#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n' + '\n'.join(choices) + '\n#endif\n'
    text = text[:location] + override + text[location:]
    anchor = 'bool llamafile_sgemm(const struct ggml_compute_params * params,'
    assert text.count(anchor) == 1
    setter = '''void ggml_cpu_sgemm_set_q8_tile(int code) {
    switch (code) {
        case 12: case 14: case 22: case 23: case 24: case 32: case 41: case 42:
            q8_tile_code = code;
            break;
        default:
            q8_tile_code = 0;
    }
}

'''
    text = text.replace(anchor, setter + anchor)
    sgemm.write_text(text)
    header = cpu_dir / 'llamafile/sgemm.h'
    text = header.read_text()
    text = text.replace('bool llamafile_sgemm(', 'void ggml_cpu_sgemm_set_q8_tile(int code);\n\nbool llamafile_sgemm(')
    header.write_text(text)
    cpu = cpu_dir / 'ggml-cpu.c'
    text = cpu.read_text()
    text = text.replace('if (nth % candidate == 0) groups = candidate;', 'if (nth % candidate == 0 && n_ids % candidate == 0) groups = candidate;')
    anchor = '            const char * cols2 = getenv("GGML_CPU_Q8_COLS2");'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''#if GGML_USE_LLAMAFILE
            const char * tile = getenv("GGML_CPU_Q8_SGEMM_TILE");
            ggml_cpu_sgemm_set_q8_tile(tile ? atoi(tile) : 0);
#endif
''' + anchor)
    # Preserve arithmetic, only assign each output to a worker near its rows.
    start = text.index('void ggml_compute_forward_mul_mat(\n')
    end = text.index('// ggml_compute_forward_mul_mat_id', start)
    body = text[start:end]
    anchor = '    const bool src1_cont = ggml_is_contiguous(src1);'
    assert body.count(anchor) == 1
    body = body.replace(anchor, anchor + '''
    struct ggml_compute_params ranked_params = *params;
    if (ggml_cpu_q8_dense_rows == 2 && !params->use_ref && src0->type == GGML_TYPE_Q8_0 && nth == 24) {
        ranked_params.ith = (ith % 6) + (ith >= 12 ? 6 : 0) + (ith % 12 >= 6 ? 12 : 0);
    }''')
    body = body.replace('llamafile_sgemm(params,', 'llamafile_sgemm(&ranked_params,')
    body = body.replace('src0->type == GGML_TYPE_Q8_0 && nr1 == 1)',
                        'src0->type == GGML_TYPE_Q8_0 && (nr1 == 1 || (ggml_cpu_q8_dense_rows == 2 && ne02 == 1 && ne03 == 1)))')
    body = body.replace('params, dst, src0->type, 1, begin, end, 0, 1);',
                        'params, dst, src0->type, 1, begin, end, 0, nr1);')
    text = text[:start] + body + text[end:]
    anchor = '        // tiled takes over if profitable for this expert (see tiled.h)'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''        if (ggml_cpu_q8_moe_schedule == 6 && !params->use_ref && !pair &&
                type == GGML_TYPE_Q8_0 && ids->ne[1] > 1) {
            int rank = ith;
            if (nth == 24) rank = (ith % 6) + (ith >= 12 ? 6 : 0) + (ith % 12 >= 6 ? 12 : 0);
            const int64_t begin = ne01 * rank / nth;
            const int64_t end = ne01 * (rank + 1) / nth;
            const void * wdata = src1->type == vec_dot_type ? src1->data : params->wdata;
            ggml_compute_forward_mul_mat_id_one_chunk(params->use_ref, dst, src0, src1, ids, cur_a,
                begin, end, 0, cne1, (const char *) src0->data + cur_a * nb02, matrix_rows,
                ggml_row_size(vec_dot_type, ne10), src1_cont, wdata);
            continue;
        }

''' + anchor)
    cpu.write_text(text)
    commands = [[item.replace(str(BASE), str(DEST)) for item in command]
                for command in json.loads((BASE / 'prepare.json').read_text())['commands']]
    (DEST / 'prepare.json').write_text(json.dumps({'base': str(BASE), 'commands': commands, 'adopted': False}, indent=2) + '\n')
    for label, command in zip(['configure', 'build'], commands):
        print(label, flush=True)
        with (DEST / (label + '.log')).open('w') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
