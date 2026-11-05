#!/usr/bin/env python3
"""Isolated AVX1 Q8 kernel and single-token MoE scheduling experiments."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
BASE = LAB / 'review-glm-q8-columns4-20261020'
DEST = LAB / 'review-q8-pipeline-20261025'

DIRECT = r'''
    // Decode has one token. Direct IDs preserve duplicate IDs and avoid the
    // per-expert mapping scan. Each output row keeps the original dot order.
    if (ggml_cpu_q8_moe_schedule && !params->use_ref && !pair &&
            type == GGML_TYPE_Q8_0 && ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
            ne12 == 1 && ne13 == 1 && ne03 == 1) {
        ggml_barrier(params->threadpool);
        const void * wdata = src1->type == vec_dot_type ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);
        ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
        int rank = ith;
        if ((ggml_cpu_q8_moe_schedule == 3 || ggml_cpu_q8_moe_schedule == 6) && nth == 24) {
            // Explicit private host layout: nodes {0..5,12..17}/{6..11,18..23}.
            // This is always a permutation; affinity only affects performance.
            rank = (ith % 6) + (ith >= 12 ? 6 : 0) + (ith % 12 >= 6 ? 12 : 0);
        }
        int groups = 1;
        if (ggml_cpu_q8_moe_schedule == 2) {
            for (int candidate = 2; candidate <= n_ids && candidate <= nth; ++candidate) {
                if (nth % candidate == 0) groups = candidate;
            }
        }
        const int group = rank % groups;
        const int worker = rank / groups;
        const int workers = nth / groups;
        const int64_t tiles = (ne01 + 15) / 16;
        const bool balanced = ggml_cpu_q8_moe_schedule >= 5;
        const int64_t begin = balanced ? ne01 * worker / workers : 16 * (tiles * worker / workers);
        const int64_t end = balanced ? ne01 * (worker + 1) / workers : MIN(16 * (tiles * (worker + 1) / workers), ne01);
        for (int id = group; id < n_ids; id += groups) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + id * ids->nb[0]);
            GGML_ASSERT(expert >= 0 && expert < n_as);
            const int64_t i11 = id % ne11;
            const char * column = (const char *) wdata +
                (src1_cont || src1->type != vec_dot_type ? i11 * row_size : i11 * nb11);
            const char * weight = (const char *) src0->data + expert * nb02;
            float * output = (float *) ((char *) dst->data + id * nb1);
            if (ggml_cpu_q8_moe_schedule == 4) {
                for (int64_t tile = ith; tile < tiles; tile += nth) {
                    for (int64_t row = tile * 16; row < MIN((tile + 1) * 16, ne01); ++row) {
                        vec_dot(ne00, output + row, 0, weight + row * nb01, 0, column, 0, 1);
                    }
                }
            } else {
                for (int64_t row = begin; row < end; ++row) {
                    vec_dot(ne00, output + row, 0, weight + row * nb01, 0, column, 0, 1);
                }
            }
        }
        return;
    }

'''

def main():
    DEST.mkdir(exist_ok=False)
    source = DEST / 'source'
    shutil.copytree(BASE / 'source', source)
    rel = Path('ggml/src/ggml-cpu')
    qfile = source / rel / 'arch/x86/quants.c'
    original = qfile.read_text()
    declaration = original[original.index('void ggml_vec_dot_q8_0_q8_0('):].split('{', 1)[0].strip()
    preamble = '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'
    preamble += 'static int ivb_q8_pipeline = 0;\nvoid ggml_cpu_q8_set_pipeline(int value) { ivb_q8_pipeline = value; }\n'
    for name in ['u4', 'u8']:
        preamble += declaration.replace('ggml_vec_dot_q8_0_q8_0', 'ivb_q8_' + name) + ';\n'
    preamble += '#endif\n\n'
    position = original.index(declaration)
    text = original[:position] + preamble + original[position:]
    anchor = declaration + ' {'
    assert text.count(anchor) == 1
    dispatch = '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'
    for value, name in [(4, 'u4'), (8, 'u8')]:
        dispatch += f'    if (ivb_q8_pipeline == {value}) {{ ivb_q8_{name}(n, s, bs, vx, bx, vy, by, nrc); return; }}\n'
    dispatch += '#endif\n'
    text = text.replace(anchor, anchor + dispatch)
    text += '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\n'
    for name in ['u4', 'u8']:
        text += (ROOT / (name + '-function.c')).read_text() + '\n'
    text += '#endif\n'
    qfile.write_text(text)
    header = source / rel / 'quants.h'
    text = header.read_text()
    anchor = next(line for line in text.splitlines() if line.startswith('void ggml_vec_dot_q8_0_q8_0('))
    text = text.replace(anchor, anchor + '\n#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)\nvoid ggml_cpu_q8_set_pipeline(int value);\n#endif')
    header.write_text(text)
    cpu = source / rel / 'ggml-cpu.c'
    text = cpu.read_text()
    text = text.replace('static const struct ggml_type_traits_cpu type_traits_cpu[',
                        'static int ggml_cpu_q8_moe_schedule = 0;\n\nstatic const struct ggml_type_traits_cpu type_traits_cpu[')
    anchor = '    if (ith == 0) {\n        // initialize matrix_row_counts'
    assert text.count(anchor) == 1
    text = text.replace(anchor, DIRECT + anchor)
    anchor = '            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''            const char * schedule = getenv("GGML_CPU_Q8_MOE_SCHEDULE");
            ggml_cpu_q8_moe_schedule = schedule ? atoi(schedule) : 0;
            if (ggml_cpu_q8_moe_schedule < 0 || ggml_cpu_q8_moe_schedule > 6) ggml_cpu_q8_moe_schedule = 0;
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__FMA__)
            const char * pipeline = getenv("GGML_CPU_Q8_PIPELINE");
            ggml_cpu_q8_set_pipeline(pipeline ? atoi(pipeline) : 0);
#endif
''' + anchor)
    cpu.write_text(text)
    patch, hashes = [], {}
    for path in [rel / 'ggml-cpu.c', rel / 'quants.h', rel / 'arch/x86/quants.c']:
        before, after = (BASE / 'source' / path).read_text(), (source / path).read_text()
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/' + str(path), tofile='b/' + str(path)))
        hashes[str(path)] = {'before': hashlib.sha256(before.encode()).hexdigest(), 'after': hashlib.sha256(after.encode()).hexdigest()}
    (DEST / 'candidate.patch').write_text(''.join(patch))
    commands = json.loads((BASE / 'prepare.json').read_text())['commands']
    commands = [[x.replace(str(BASE), str(DEST)) for x in command] for command in commands]
    # Pin the intended ISA independently of the host running CMake.
    commands[0] += ['-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=OFF', '-DGGML_FMA=OFF',
                    '-DGGML_F16C=ON', '-DGGML_BMI2=OFF', '-DGGML_AVX512=OFF', '-DCMAKE_C_FLAGS=-march=ivybridge',
                    '-DCMAKE_CXX_FLAGS=-march=ivybridge']
    (DEST / 'prepare.json').write_text(json.dumps({'hashes': hashes, 'commands': commands, 'adopted': False}, indent=2) + '\n')
    for label, command in zip(['configure', 'build'], commands):
        print(label, flush=True)
        with (DEST / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    main()
