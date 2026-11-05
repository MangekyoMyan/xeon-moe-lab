#!/usr/bin/env python3
"""Complete the isolated candidate and record its final reproducible diff."""
import difflib
import hashlib
import json
from pathlib import Path

lab = Path(__file__).resolve().parent
root = lab / 'review-numa-rows-20261020'
out = lab / 'results/optimization-numa-rows-20261020'
source = root / 'source'
p = source / 'ggml/src/ggml-cpu/numa-replica.cpp'
s = p.read_text().replace('\\\\n', '\\n')
old = '''static bool shard_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_CPU_NUMA_SHARD");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}'''
new = '''static int shard_mode() {
    static const int mode = [] {
        const char * env = std::getenv("GGML_CPU_NUMA_SHARD");
        if (env && std::strcmp(env, "1") == 0) return 1;
        if (env && std::strcmp(env, "2") == 0) return 2;
        return 0;
    }();
    return mode;
}

static bool shard_enabled() { return shard_mode() != 0; }'''
assert s.count(old) == 1
s = s.replace(old, new).replace('if (context->shard && eligible(tensor)) {',
                               'if (shard_mode() == 1 && eligible(tensor)) {')
old = '''                context->calls[node].fetch_add(1, std::memory_order_relaxed);'''
new = '''                if (context->calls[node].fetch_add(1, std::memory_order_relaxed) < 12) {
                    std::fprintf(stderr, "NUMA_SHARD_WORK ith=%d nth=%d node=%u\\n", params->ith, params->nth, node);
                }'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
p = source / 'ggml/src/ggml-cpu/ggml-cpu.c'
s = p.read_text()
start = s.index('static void ggml_compute_forward_mul_mat_impl(')
end = s.index('// ggml_compute_forward_mul_mat_id', start)
part = s[start:end]
assert part.count('    if (src1_cont) {') == 1
part = part.replace('    if (src1_cont) {', '    if (row_start < 0 && src1_cont) {')
part = part.replace('#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {',
                    '#if GGML_USE_LLAMAFILE\n    if (row_start < 0 && src1->type != vec_dot_type) {')
p.write_text(s[:start] + part + s[end:])

p = lab / 'run_cpu_ab_numa_rows.py'
s = p.read_text()
s = s.replace("if key in ('GGML_CPU_NUMA_REPLICA', 'GGML_CPU_NUMA_SHARD', 'GGML_CPU_MOE_PAIR')",
              "if key in ('GGML_CPU_NUMA_REPLICA', 'GGML_CPU_MOE_PAIR')")
s = s.replace('        overrides[key] = setting', """        if key == 'GGML_CPU_NUMA_SHARD' and setting not in ('0', '1', '2'):
            parser.error('GGML_CPU_NUMA_SHARD must be 0, 1 or 2.')
        overrides[key] = setting""")
p.write_text(s)
p = out / 'numa-shard-probe.cpp'
s = p.read_text().replace('        ggml_backend_buffer_clear(replica_buffer, 0x37);\n', '')
s = s.replace('        std::vector<unsigned char> values(bytes), restored(bytes);',
              '        ggml_backend_buffer_clear(replica_buffer, 0x37);\n        std::vector<unsigned char> values(bytes), restored(bytes);')
s = s.replace('        ggml_backend_tensor_get(replica, restored.data(), 0, bytes);',
              '''        ggml_backend_tensor_set(replica, values.data() + bytes / 2 - 17, bytes / 2 - 17, 34);
        ggml_backend_tensor_get(replica, restored.data(), 0, bytes);''')
s = s.replace('{{1, 0}, {1, 6}, {6, 0}, {6, 6}, {12, 0}, {24, 0}}',
              '{{1, 0}, {1, 6}, {3, 0}, {6, 0}, {6, 6}, {12, 0}, {24, 0}}')
p.write_text(s)
record = json.loads((out / 'experiment.json').read_text())
patch = ''
for relative in record['files']:
    before = (lab / 'llama-private-prefetch' / relative).read_text()
    after = (source / relative).read_text()
    record['files'][relative]['after'] = hashlib.sha256(after.encode()).hexdigest()
    patch += ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/' + relative, tofile='b/' + relative))
record['shard_modes'] = {'0': 'disabled', '1': 'row halves local to nodes 0/1 plus fixed rows',
                         '2': 'interleaved single copy plus identical fixed-row schedule; localization control'}
(root / 'candidate.patch').write_text(patch)
for p in (root / 'prepare.json', out / 'experiment.json'):
    p.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'refined': True, 'source_files': list(record['files'])}))
