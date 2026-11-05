#!/usr/bin/env python3
"""Prepare optional two-column tails, dense column reuse, and Q8 MoE pairing."""
import argparse
import difflib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-q8-pipeline-20261025'
DEST = LAB / 'review-q8-reuse-20261025'

DENSE = r'''
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
                if (ggml_cpu_q8_dense_cols && !params->use_ref && type == GGML_TYPE_Q8_0 && num_rows_per_vec_dot == 1) {
                    int columns = 0;
                    if (ggml_cpu_q8_dense_cols == 4 && ir1 + 3 < ir1_end && ir1 + 3 < iir1 + blck_1 && i11 + 3 < ne11) {
                        columns = 4;
                    } else if (ir1 + 1 < ir1_end && ir1 + 1 < iir1 + blck_1 && i11 + 1 < ne11) {
                        columns = 2;
                    }
                    if (columns) {
                        const void * inputs[4];
                        for (int c = 0; c < columns; ++c) inputs[c] = src1_col + c * src1_col_stride;
                        for (int64_t row = iir0; row < MIN(iir0 + blck_0, ir0_end); ++row) {
                            float sums[4];
                            if (columns == 4) ggml_vec_dot_q8_0_q8_0_cols4(ne00, sums, src0_row + row * nb01, inputs);
                            else ggml_vec_dot_q8_0_q8_0_cols2(ne00, sums, src0_row + row * nb01, inputs);
                            for (int c = 0; c < columns; ++c) {
                                float * output = (float *) ((char *) dst_col + c * nb1);
                                output[row] = sums[c];
                            }
                        }
                        ir1 += columns - 1;
                        continue;
                    }
                }
#endif
'''

def block_end(text, start):
    begin = text.index('{', start)
    depth = 1
    pos = begin + 1
    while depth:
        depth += (text[pos] == '{') - (text[pos] == '}')
        pos += 1
    return pos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(exist_ok=False)
    source = DEST / 'source'
    shutil.copytree(BASE / 'source', source)
    rel = Path('ggml/src/ggml-cpu')
    qfile = source / rel / 'arch/x86/quants.c'
    text = qfile.read_text()
    start = text.index('void ggml_vec_dot_q8_0_q8_0_cols4(')
    end = block_end(text, start)
    fn = text[start:end]
    fn = fn.replace('q8_0_cols4(', 'q8_0_cols2(').replace('[4]', '[2]').replace('c < 4', 'c < 2')
    fn = fn.replace('{ vy[0], vy[1], vy[2], vy[3] }', '{ vy[0], vy[1] }')
    text = text[:end] + '\n\n' + fn + text[end:]
    qfile.write_text(text)
    header = source / rel / 'quants.h'
    text = header.read_text()
    declaration = next(line for line in text.splitlines() if line.startswith('void ggml_vec_dot_q8_0_q8_0_cols4('))
    text = text.replace(declaration, declaration + '\n' + declaration.replace('q8_0_cols4(', 'q8_0_cols2(').replace('[4]', '[2]'))
    header.write_text(text)
    cpu = source / rel / 'ggml-cpu.c'
    text = cpu.read_text()
    anchor = 'static int ggml_cpu_q8_moe_schedule = 0;'
    text = text.replace(anchor, anchor + '\nstatic bool ggml_cpu_q8_cols2 = false;\nstatic int ggml_cpu_q8_dense_cols = 0;\nstatic bool ggml_cpu_q8_moe_pair = false;')
    anchor = '                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n' + DENSE)
    start = text.index('            if (!use_ref && type == GGML_TYPE_Q8_0 && ids->ne[1] > 1) {')
    loop = text.index('                for (; ir1 + 3', start)
    end = block_end(text, loop)
    tail = text[loop:end]
    tail = tail.replace('ir1 + 3', 'ir1 + 1').replace('ir1 += 4', 'ir1 += 2')
    tail = tail.replace('[4]', '[2]').replace('c < 4', 'c < 2').replace('q8_0_cols4(', 'q8_0_cols2(')
    tail = '\n'.join('    ' + line for line in tail.splitlines())
    text = text[:end] + '\n                if (ggml_cpu_q8_cols2) {\n' + tail + '\n                }' + text[end:]
    anchor = 'ggml_cpu_q8_moe_schedule && !params->use_ref && !pair &&'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'ggml_cpu_q8_moe_schedule && !params->use_ref &&')
    anchor = 'vec_dot(ne00, output + row, 0, weight + row * nb01, 0, column, 0, 1);'
    assert text.count(anchor) == 2
    pair = '''
                        if (pair) {
                            const struct ggml_tensor * paired_weight = pair->src[0];
                            float * paired_output = (float *) ((char *) pair->data + id * pair->nb[1]);
                            const char * paired_row = (const char *) paired_weight->data + expert * paired_weight->nb[2] + row * paired_weight->nb[1];
                            vec_dot(ne00, paired_output + row, 0, paired_row, 0, column, 0, 1);
                        }'''
    text = text.replace(anchor, anchor + pair)
    anchor = 'a->src[0]->type != GGML_TYPE_IQ2_XXS || b->src[0]->type != GGML_TYPE_IQ2_XXS ||'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''a->src[0]->type != b->src[0]->type ||
        !(a->src[0]->type == GGML_TYPE_IQ2_XXS || (ggml_cpu_q8_moe_pair && a->src[0]->type == GGML_TYPE_Q8_0)) ||''')
    anchor = '            const char * schedule = getenv("GGML_CPU_Q8_MOE_SCHEDULE");'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''            const char * cols2 = getenv("GGML_CPU_Q8_COLS2");
            const char * dense_cols = getenv("GGML_CPU_Q8_DENSE_COLS");
            const char * q8_pair = getenv("GGML_CPU_Q8_MOE_PAIR");
            ggml_cpu_q8_cols2 = cols2 && strcmp(cols2, "1") == 0;
            ggml_cpu_q8_dense_cols = dense_cols ? atoi(dense_cols) : 0;
            if (ggml_cpu_q8_dense_cols != 2 && ggml_cpu_q8_dense_cols != 4) ggml_cpu_q8_dense_cols = 0;
            ggml_cpu_q8_moe_pair = q8_pair && strcmp(q8_pair, "1") == 0;
''' + anchor)
    cpu.write_text(text)
    patch = []
    for path in [rel / 'ggml-cpu.c', rel / 'quants.h', rel / 'arch/x86/quants.c']:
        before, after = (BASE / 'source' / path).read_text(), (source / path).read_text()
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/' + str(path), tofile='b/' + str(path)))
    (DEST / 'incremental.patch').write_text(''.join(patch))
    commands = [[s.replace(str(BASE), str(DEST)) for s in command] for command in json.loads((BASE / 'prepare.json').read_text())['commands']]
    (DEST / 'prepare.json').write_text(json.dumps({'base': str(BASE), 'commands': commands, 'adopted': False}, indent=2) + '\n')
    if args.build:
        for label, command in zip(['configure', 'build'], commands):
            print(label, flush=True)
            with (DEST / (label + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
