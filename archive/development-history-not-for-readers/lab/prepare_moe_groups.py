#!/usr/bin/env python3
"""Prepare an isolated expert-group scheduling experiment."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent
root = lab / 'review-moe-groups-20261020'
out = lab / 'results/optimization-numa-rows-20261020'
root.mkdir(exist_ok=False)
source = root / 'source'
shutil.copytree(lab / 'llama-private-prefetch', source)
relative = 'ggml/src/ggml-cpu/ggml-cpu.c'
before = (source / relative).read_text()
after = before.replace('#define MMID_MATRIX_ROW', 'static bool ggml_cpu_moe_groups = false;\n\n#define MMID_MATRIX_ROW', 1)
start = after.index('static void ggml_compute_forward_mul_mat_id(')
end = after.index('\n/////////////////////////////////', start)
part = after[start:end]
anchor = '''    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {'''
assert part.count(anchor) == 1
part = part.replace(anchor, '''    ggml_barrier(params->threadpool);

    int groups = 0;
    if (ggml_cpu_moe_groups && !params->use_ref && type == GGML_TYPE_IQ2_XXS && ids->ne[1] == 1) {
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            if (matrix_row_counts[cur_a] > 1) {
                groups = 0;
                break;
            }
            groups += matrix_row_counts[cur_a] != 0;
        }
        groups = MIN(groups, nth);
    }
    const int group = groups ? ((ith + 1) * groups - 1) / nth : 0;
    const int group_start = groups ? nth * group / groups : 0;
    const int group_size = groups ? nth * (group + 1) / groups - group_start : 0;
    int active_rank = 0;

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {''')
anchor = '''        // tiled takes over if profitable for this expert (see tiled.h)
        if (!pair && ggml_compute_forward_mul_mat_id_tiled'''
assert part.count(anchor) == 1
part = part.replace(anchor, '''        if (groups && active_rank++ % groups != group) {
            continue;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        if (!groups && !pair && ggml_compute_forward_mul_mat_id_tiled''')
anchor = '''        const int64_t nr1 = cne1;

        int chunk_size = 16;'''
assert part.count(anchor) == 1
part = part.replace(anchor, '''        const int64_t nr1 = cne1;

        if (groups) {
            const int rank = ith - group_start;
            const int64_t tiles = (nr0 + 15) / 16;
            const int64_t row_start = 16 * (tiles * rank / group_size);
            const int64_t row_end = MIN(16 * (tiles * (rank + 1) / group_size), nr0);
            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a, row_start, row_end, 0, nr1,
                src0_cur, matrix_rows, row_size, src1_cont, wdata);
            if (pair) {
                const struct ggml_tensor * weight = pair->src[0];
                ggml_compute_forward_mul_mat_id_one_chunk(
                    pair, weight, src1, ids, cur_a, row_start, row_end, 0, nr1,
                    (const char *) weight->data + cur_a * weight->nb[2],
                    matrix_rows, row_size, src1_cont, wdata);
            }
            continue;
        }

        int chunk_size = 16;''')
after = after[:start] + part + after[end:]
anchor = '''            env = getenv("GGML_CPU_TILED_MM_FORCE");
            if (env != NULL && atoi(env) != 0) {
                ggml_cpu_moe_pair = false;'''
assert after.count(anchor) == 1
after = after.replace(anchor, '''            env = getenv("GGML_CPU_MOE_GROUPS");
            ggml_cpu_moe_groups = (env != NULL && strcmp(env, "1") == 0);
            env = getenv("GGML_CPU_TILED_MM_FORCE");
            if (env != NULL && atoi(env) != 0) {
                ggml_cpu_moe_groups = false;
                ggml_cpu_moe_pair = false;''')
(source / relative).write_text(after)
(root / 'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
    fromfile='a/' + relative, tofile='b/' + relative)))
(root / 'prepare.json').write_text(json.dumps({'baseline': str(lab / 'llama-private-prefetch'),
    'file': relative, 'before': hashlib.sha256(before.encode()).hexdigest(),
    'after': hashlib.sha256(after.encode()).hexdigest(), 'adopted': False,
    'scope': 'IQ2_XXS single-token unique-ID MoE only. Split workers into contiguous groups, assign active experts round-robin to groups, reuse existing dot kernels and barriers. No weight placement or numeric changes.'}, indent=2) + '\n')
p = lab / 'run_cpu_ab_numa_rows.py'
s = p.read_text().replace("'GGML_CPU_NUMA_SHARD', 'GGML_CPU_MOE_PAIR'", "'GGML_CPU_NUMA_SHARD', 'GGML_CPU_MOE_GROUPS', 'GGML_CPU_MOE_PAIR'")
s = s.replace("('GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR')", "('GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_GROUPS', 'GGML_CPU_MOE_PAIR')")
p.write_text(s)
probe = (lab / 'moe-pair-probe.cpp').read_text().replace('{1, 4, 12}', '{1, 2, 4, 6, 12, 24}')
(out / 'moe-groups-probe.cpp').write_text(probe)
print(json.dumps({'prepared': str(root)}))
