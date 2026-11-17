#!/usr/bin/env python3
"""Independent read-only audit of published d4d50b6 parallel-pilot evidence.

Only this auditor is executed. No supplied code, ELF, model, GPU, API or engine
is executed or imported. Inputs are JSON, UTF-8 source/logs and little-endian
numeric arrays. NumPy reductions are independent from the supplied analyzer.
This audits saved observations; it is not a fresh hardware benchmark.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument('--repo', type=Path, default=Path('/workspace/scratch/1f624f9ac929/xeon-moe-parallel-d4d50b6'))
AP.add_argument('--output', type=Path, default=Path(__file__).with_name('INDEPENDENT_EVIDENCE_AUDIT.json'))
AP.add_argument('--prior-zip', type=Path, default=Path('/workspace/scratch/1f624f9ac929/deliverables/xeon_moe_gpu_workspace_review_2026-11-17.zip'))
A = AP.parse_args()
ROOT = A.repo / 'archive/review-handoffs-not-for-readers/2026-11-17-parallel-pilot'
PRIOR = A.repo / 'archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace/pilot-capture/runs/20261117T035402Z-U1024-d22b3e04'
COMMIT = 'd4d50b651ff7754ee2189228762468fc02971bfa'
checks = collections.Counter()
failures = []
def check(label, passed, detail=None):
    checks[label.split(':')[0]] += 1
    if not bool(passed):
        failures.append({'check': label, 'detail': detail})
def js(p):
    return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
def close(a, b, rel=2e-11, absolute=1e-11):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=absolute)
def equality(a, b):
    return a.shape == b.shape and a.tobytes() == b.tobytes()
def f32(p, count=None):
    a = np.fromfile(p, dtype='<f4')
    check('array:finite:' + str(p.relative_to(ROOT)), np.isfinite(a).all())
    if count is not None:
        check('array:size:' + str(p.relative_to(ROOT)), a.size == count, [a.size, count])
    return a
def describe(a):
    a = np.asarray(a, dtype=np.float64)
    return {'count': a.size, 'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'p10': float(np.quantile(a, .1)),
            'p90': float(np.quantile(a, .9)), 'max': float(a.max())}
def metric(a, ref):
    aa, rr = np.asarray(a, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    diff = aa - rr
    norm = float(np.sum(rr * rr))
    se = float(np.sum(diff * diff))
    return {'nmse': se / norm if norm else None, 'sum_squared_error': se,
            'reference_norm2': norm, 'output_norm2': float(np.sum(aa * aa)),
            'max_abs': float(np.max(np.abs(diff))), 'count': aa.size,
            'byte_equal': equality(a, ref)}
def compare_report(actual, report, name):
    """Compare shared numeric fields; descriptions are not assertions."""
    for k, v in actual.items():
        if k not in report:
            continue
        t = report[k]
        if isinstance(v, dict) and isinstance(t, dict):
            compare_report(v, t, name + '/' + k)
        elif isinstance(v, (bool, str)) or v is None:
            check('reported:' + name + '/' + k, v == t, [v, t])
        elif isinstance(v, (int, float)):
            check('reported:' + name + '/' + k, close(v, t), [v, t])
        elif isinstance(v, list) and isinstance(t, list):
            check('reported:' + name + '/' + k, len(v) == len(t) and np.allclose(v, t, rtol=2e-11, atol=1e-11), {'actual_length': len(v), 'reported_length': len(t)})
def weighted(v, w):
    """F32 multiply and left-to-right F32 add, without contraction."""
    z = np.multiply(v[0], w[0], dtype=np.float32)
    for i in range(1, 6):
        term = np.multiply(v[i], w[i], dtype=np.float32)
        z = np.add(z, term, dtype=np.float32)
    return z
def source_info(p):
    return {'path': str(p.relative_to(A.repo)), 'bytes': p.stat().st_size, 'sha256': sha(p)}

git_head = subprocess.check_output(['git', '-C', str(A.repo), 'rev-parse', 'HEAD'], text=True).strip()
check('provenance:commit', git_head == COMMIT, git_head)
result = {'source_commit': git_head, 'method': __doc__, 'numpy_version': np.__version__}

# Published payload and pre-existing handoff: hash data, never trust a flag alone.
manifest = js(ROOT / 'FILES_SHA256.json')
hash_paths = collections.defaultdict(list)
for name, pin in manifest.items():
    p = ROOT / name
    check('manifest:exists:' + name, p.is_file())
    if p.is_file():
        digest = sha(p)
        hash_paths[digest].append(name)
        check('manifest:hash:' + name, digest == pin['sha256'])
        check('manifest:size:' + name, p.stat().st_size == pin['bytes'])
actual_files = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
check('manifest:complete-file-set', actual_files == set(manifest) | {'FILES_SHA256.json'})
mapping = js(ROOT / 'RAW_MAPPING.json')
check('rawmapping:complete-file-set', {x['shared'] for x in mapping} == set(manifest) - {'RAW_MAPPING.json'})
for rec in mapping:
    check('rawmapping:hash:' + rec['shared'], sha(ROOT / rec['shared']) == rec['sha256'])
    check('rawmapping:byte-identity-claim:' + rec['shared'], rec['byte_identical'] is True)
incoming = ROOT / 'incoming/gpu_workspace_review_2026-11-17'
im = js(incoming / 'FILE_MANIFEST.json')
for name, pin in im['files'].items():
    check('incoming:hash:' + name, sha(incoming / name) == pin['sha256'])
    check('incoming:bytes:' + name, (incoming / name).stat().st_size == pin['bytes'])
check('incoming:payload-count', len(im['files']) == im['payload_files'] == 29)
zin = js(ROOT / 'INTAKE.json')
zip_result = {'available': A.prior_zip.is_file()}
if A.prior_zip.is_file():
    zip_result.update({'bytes': A.prior_zip.stat().st_size, 'sha256': sha(A.prior_zip)})
    check('incoming:prior-zip-size', zip_result['bytes'] == 130571)
    check('incoming:prior-zip-sha', zip_result['sha256'] == zin['sha256'] == '4c5cfaff5437b9e31e68be95ecca9b221e8dfcd9d7b287972804d40d4c38bf7f')
    with zipfile.ZipFile(A.prior_zip) as z:
        members = [x for x in z.namelist() if not x.endswith('/')]
        zip_result['members'] = len(members)
        check('incoming:zip-members', len(members) == 30)
        for name in members:
            rel = name.split('/', 1)[1]
            check('incoming:zip-byte-equal:' + rel, z.read(name) == (incoming / rel).read_bytes())
result['published_payload'] = {'files': len(manifest), 'total_bytes': sum(x['bytes'] for x in manifest.values()), 'raw_mapping_entries': len(mapping), 'prior_zip': zip_result}

# Every run's own original manifest, launch and runtime-provider metadata.
run_dirs = sorted(ROOT.glob('*/runs/*'))
runs = {}
run_provenance = []
for d in run_dirs:
    r = js(d / 'RUN.json'); runs[d.name] = r
    launch = js(d / 'LAUNCH_MANIFEST.json')
    original = js(d / 'RESULTS_SHA256.json')
    for name, digest in original.items():
        if name == 'server-config.json':
            encoded = (json.dumps({'command': launch['command'], 'environment': launch['environment']}, indent=2) + '\n').encode()
            check('runhash:' + d.name + '/' + name, hashlib.sha256(encoded).hexdigest() == digest)
        else:
            local = 'guard-health.json' if name == 'health.json' else name
            check('runhash:' + d.name + '/' + name, sha(d / local) == digest)
    expected = js(d / 'PROVIDER_EXPECTED.json')
    check('provider:' + d.name + '/symbols', r['providers'] == expected['providers'])
    check('provider:' + d.name + '/backends', r['active_backend_providers'] == expected['active_backend_providers'])
    check('provider:' + d.name + '/GPU', launch['environment']['CUDA_VISIBLE_DEVICES'] == expected['CUDA_VISIBLE_DEVICES'])
    build = js(d / 'BUILD.json')
    check('build:' + d.name + '/launch-build-hash', sha(d / 'BUILD.json') == launch['build_sha256'])
    check('build:' + d.name + '/returncode', build['returncode'] == 0)
    source_resolved = {}
    for name, digest in build['source_sha256'].items():
        candidates = hash_paths.get(digest, [])
        check('build:' + d.name + '/source/' + name, bool(candidates), digest)
        source_resolved[name] = candidates
    check('build:' + d.name + '/runner-published', bool(hash_paths.get(launch['runner_sha256'])), launch['runner_sha256'])
    health = js(d / 'guard-health.json')
    check('health:' + d.name + '/rc', health['child_returncode'] == (0 if r['passed'] else 1))
    check('health:' + d.name + '/end-time', health['finished_unix'] >= health['started_unix'])
    lib = build['engine_libraries']['diagnostic']
    by_path = {v['path']: v['sha256'] for v in lib.values()}
    important = {'llama_decode', 'llama_get_logits_ith', 'ggml_backend_cpu_init', 'ggml_backend_sched_graph_compute_async'}
    for pr in r['providers']:
        if pr['symbol'] in important:
            check('provider:' + d.name + '/source-path/' + pr['symbol'], pr['path'] in by_path)
    memory = []
    for ph in r['memory_phases']:
        cu, nv = ph['cuda'], ph['nvml_cli']
        check('memory:' + d.name + '/' + ph['phase'] + '/query-order', cu['begin_ns'] < cu['end_ns'] <= nv['begin_ns'] < nv['end_ns'])
        check('memory:' + d.name + '/' + ph['phase'] + '/nvml-status', nv['status'] == 0)
        memory.append({'phase': ph['phase'], 'cuda_free_MiB': cu['free_bytes'] / 2**20, 'sequential_queries': True})
    run_provenance.append({'run': str(d.relative_to(ROOT)), 'passed': r['passed'], 'libraries_saved_hashes': lib, 'source_resolved': source_resolved, 'memory_phases': memory, 'health_returncode': health['child_returncode']})
result['runs_provenance'] = run_provenance
engine = js(ROOT / 'integrated/ENGINE_BUILD.json')
for name, digest in engine['source_sha256'].items():
    check('engine:source:' + name, sha(ROOT / 'integrated' / name) == digest)
for d in ROOT.glob('integrated/runs/*'):
    launch, build = js(d / 'LAUNCH_MANIFEST.json'), js(d / 'BUILD.json')
    check('engine:launch-pin:' + d.name, sha(ROOT / 'integrated/ENGINE_BUILD.json') == launch['engine_build_sha256'])
    check('engine:driver-library:' + d.name, build['engine_libraries']['diagnostic']['libllama.so.0.8.0']['sha256'] == engine['libllama_sha256'])

# Plan/native inventory and explicit Gpre/Gpost observations.
plan_lines = (ROOT / 'integrated/capacity.plan').read_text().splitlines()
hdr = plan_lines[0].split()
selected = {}
for line in plan_lines[1:]:
    f = line.split(); layer = int(f[0]); ids = list(map(int, f[5:])); selected[layer] = ids
    check('plan:count:' + str(layer), len(ids) == int(f[4]) and len(set(ids)) == len(ids))
check('plan:unchanged-copy', (ROOT / 'integrated/capacity.plan').read_bytes() == (ROOT / 'replay/capacity.plan').read_bytes())
active = [l for l in range(37) if selected[l]]
empty = [l for l in range(37) if not selected[l]]
check('plan:shape', len(active) == 23 and len(empty) == 14 and sum(map(len, selected.values())) == 35)
inventory = runs['20261117T054816Z-CSP-U1024-2ff29532']['native_expert_inventory']
native_bytes = sum(len(selected[x['layer']]) * x['native_triplet_bytes'] for x in inventory)
check('plan:native-bytes', native_bytes == int(hdr[-1]) == 475453440)
for inv in inventory:
    check('plan:host:' + str(inv['layer']), inv['host'] == (inv['layer'] < 37))
    check('plan:projection-bytes:' + str(inv['layer']), inv['native_triplet_bytes'] == sum(p['native_expert_bytes'] for p in inv['projections']))
guards = []
for d in [ROOT / 'replay/runs/20261117T053052Z-U1024-1563d936', ROOT / 'integrated/runs/20261117T054816Z-CSP-U1024-2ff29532']:
    # Native progress logs can contain non-UTF8 byte fragments; exact bytes are
    # already hashed above, and the admission lines themselves are ASCII.
    log = (d / 'server.stderr.log').read_text(errors='replace')
    pre = re.search(r'capacity-bank-Gpre free=(\d+) total=(\d+) payload=(\d+) margin=(\d+)', log)
    post = re.search(r'capacity-bank-Gpost free=(\d+) total=(\d+) margin=(\d+)', log)
    check('admission:' + d.name + '/logged', pre is not None and post is not None)
    a, b = list(map(int, pre.groups())), list(map(int, post.groups()))
    check('admission:' + d.name + '/Gpre', a[0] >= a[2] + a[3])
    check('admission:' + d.name + '/Gpost', b[0] >= b[2])
    guards.append({'run': d.name, 'Gpre_free_MiB': a[0] / 2**20, 'Gpre_required_MiB': (a[2] + a[3]) / 2**20, 'Gpost_free_MiB': b[0] / 2**20, 'Gpost_margin_MiB': b[2] / 2**20})
result['plan'] = {'sha256': sha(ROOT / 'integrated/capacity.plan'), 'native_experts': 35, 'native_bytes': native_bytes, 'active_cpu_layers': active, 'empty_cpu_layers': empty, 'guards': guards}

# Baseline pin and joint-ID activation capture, including preserved failed runs.
for pinpath in [ROOT / 'capture/BASELINE_PIN.json', ROOT / 'integrated/BASELINE_PIN.json']:
    for name, digest in js(pinpath)['files'].items():
        p = PRIOR / ('guard-health.json' if name == 'health.json' else name)
        check('baselinepin:' + str(pinpath.relative_to(ROOT)) + '/' + name, sha(p) == digest)
baseline = js(PRIOR / 'text_cache_budget_ja.json')
base_records = baseline['T1_trace']['records']
cd = ROOT / 'capture/runs/20261117T052657Z-U1024-3483cb75'
capture = js(cd / 'CAPTURE.json')
trace = js(cd / 'text_cache_budget_ja.json')
records = trace['T1_trace']['records']
check('capture:all333-equal', len(records) == 333 and records == base_records[:333])
check('capture:full-five-row-logits-equal', (cd / 'text_cache_budget_ja.f32').read_bytes() == (PRIOR / 'text_cache_budget_ja.f32').read_bytes())
f32(cd / 'text_cache_budget_ja.f32', 5 * 129280)
check('capture:nine-calls', len(trace['T1']) == 9 and len(trace['continuation_inputs']) == 9)
planned = js(ROOT / 'CAPTURE_CELLS.json')['cells']
capture_details = []
for c, p in zip(capture['cells'], planned):
    i = c['cell']; lab = str(i)
    check('capture:metadata:' + lab, c['cell'] == p['cell'] and c['call'] == p['decode_call'] and c['layer'] == p['layer'] and c['ids'] == p['ordered_ids'])
    check('capture:state:' + lab, c['flags'] == 0 and c['seen'] == 7)
    check('capture:record:' + lab, c['route_record'] == records[p['source_record_ordinal']])
    ids = np.fromfile(cd / c['ids_file'], dtype='<i4').tolist()
    check('capture:binary-ids:' + lab, ids == c['ids'])
    slots = [k for k, eid in enumerate(ids) if eid in selected[c['layer']]]
    check('capture:H-and-slots:' + lab, slots == p['gpu_slots'] and len(slots) == p['H_under_original_candidate'])
    check('capture:projection-types:' + lab, [x['type'] for x in c['projections']] == p['projection_types'])
    for key, count in [('x_file', 5120), ('down_file', 30720), ('routing_weights_file', 6)]:
        f32(cd / c[key], count)
    check('capture:weight-readback:' + lab, c['routing_weight_readback'] and c['weights_ready_after_gate'])
    capture_details.append({'cell': i, 'layer': c['layer'], 'call': c['call'], 'ids': ids, 'H': len(slots), 'gpu_slots': slots, 'projections': p['projection_types']})
failed_capture = []
for name, expected_count in [('20261117T051531Z-U1024-4725007c', 37), ('20261117T051936Z-U1024-9d4e6e9e', 111)]:
    r = runs[name]; rs = r['failure_trace']['records']
    check('failure:capture-count:' + name, len(rs) == expected_count and not r['passed'])
    diffs = [i for i, (a, b) in enumerate(zip(rs, base_records)) if a != b]
    iddiff = [i for i, (a, b) in enumerate(zip(rs, base_records)) if a['ids'] != b['ids']]
    failed_capture.append({'run': name, 'records': len(rs), 'all_field_changed_ordinals': diffs, 'ids_changed_ordinals': iddiff, 'capture_failure': r['capture_failure'], 'error': r['error']})
check('failure:second-changed32', len(failed_capture[1]['ids_changed_ordinals']) == 32)
result['capture'] = {'cells': capture_details, 'matching_prior_route_records': len(records), 'matching_prior_logit_rows': 5, 'callback_timing_valid': capture['callback_timing_valid'], 'call5_8_callback_free_intermediate_identity_proven': capture['call5_8_intermediate_callback_free_bit_identity_proven'], 'failed_runs': failed_capture}

# Operator outputs and all first-use/timed timestamps, independent reductions.
rd = ROOT / 'replay/runs/20261117T053052Z-U1024-1563d936'
rp = js(rd / 'REPLAY.json'); rpa = js(ROOT / 'REPLAY_ANALYSIS.json')
check('replay:layers', rp['private_initialized_layers'] == active and rp['empty_layers_bypassed'] == empty)
check('replay:bank', rp['bank_native_bytes'] == rp['bank_native_bytes_verified'] == native_bytes and rp['bank_allocated_bytes'] == 475457024)
for key in ['O_worker_receipt', 'CSP_worker_receipt']:
    rec = rp[key]
    check('replay:workers:' + key, len(rec) == 12 and len({x['tid'] for x in rec}) == 12 and all(x['worker'] == x['cpu'] == x['allowed_cpu'] for x in rec) and {x['cpu'] for x in rec} == set(range(12)))
replay_out = []
for cell in rp['cells']:
    i = cell['cell']; label = f'cell{i}'
    check('replay:cell-json:' + label, cell == js(rd / (label + '.json')))
    out = {mode: f32(rd / f'{label}-{mode}.f32', 30720).reshape(6, 5120) for mode in 'OCSP'}
    cap = f32(cd / f'{label}-down.f32', 30720).reshape(6, 5120)
    w = f32(cd / f'{label}-weights.f32', 6)
    wout = {m: weighted(v, w) for m, v in out.items()}
    got = {'O_C': metric(out['O'], out['C']), 'S_P': metric(out['S'], out['P']),
           'O_capture': metric(out['O'], cap), 'P_O': metric(out['P'], out['O']),
           'P_O_weighted': metric(wout['P'], wout['O']),
           'S_P_weighted_byte_equal': equality(wout['S'], wout['P'])}
    compare_report(got, cell['correctness'], label + '/correctness')
    cold = []
    for k in range(6):
        m = metric(out['P'][k], out['O'][k]); compare_report(m, cell['correctness']['per_slot_P_O'][k], label + '/slot' + str(k))
        if k not in cell['gpu_slots']:
            check('replay:cold-slot-byte-equal:' + label + '/' + str(k), m['byte_equal']); cold.append(k)
    check('replay:O-C-capture:' + label, got['O_C']['byte_equal'] and got['O_capture']['byte_equal'])
    check('replay:S-P:' + label, got['S_P']['byte_equal'])
    ref_out = None
    if (rd / f'{label}-reference.f32').is_file():
        ref = f32(rd / f'{label}-reference.f32', 30720).reshape(6, 5120)
        wr = weighted(ref, w)
        ref_out = {m: metric(out[m], ref) for m in 'OP'}
        ref_out.update({m + '_weighted': metric(wout[m], wr) for m in 'OP'})
        compare_report(ref_out, cell['reference'], label + '/reference')
    samples = cell['samples']; first = [x for x in samples if x['first_use']]; timed = [x for x in samples if not x['first_use']]
    check('replay:sample-count:' + label, len(samples) == 52 and len(first) == 4 and len(timed) == 48)
    check('replay:first-use-order:' + label, ''.join(x['mode'] for x in first) == 'OCSP')
    orders = []
    by_block = []
    for k in range(12):
        ss = [x for x in timed if x['block'] == k]
        orders.append(''.join(x['mode'] for x in ss)); by_block.append({x['mode']: (x['output_ready_ns'] - x['input_ready_ns']) / 1e6 for x in ss})
    check('replay:balanced-order:' + label, orders == ['OCPS', 'CSOP', 'SPCO', 'POSC'] * 3)
    for j, s in enumerate(samples):
        a, b = s['input_ready_ns'], s['output_ready_ns']
        check('replay:time-positive:' + label + '/' + str(j), a < b and a <= s['cpu_begin_ns'] <= s['cpu_end_ns'] <= b)
        expected_h = cell['matched_H'] if s['mode'] in 'SP' else 0
        check('replay:execution-H:' + label + '/' + str(j), s['executed_gpu_H'] == expected_h and s['matched_H'] == cell['matched_H'])
        check('replay:GPU-calls:' + label + '/' + str(j), s['gpu_operations'] == (5 if expected_h else 0))
        if expected_h:
            check('replay:GPU-join-order:' + label + '/' + str(j), a <= s['partition_end_ns'] <= s['submit_end_ns'] <= s['join_begin_ns'] <= s['join_end_ns'] <= b)
            check('replay:serial-vs-parallel-order:' + label + '/' + str(j), s['join_end_ns'] <= s['cpu_begin_ns'] if s['mode'] == 'S' else s['cpu_end_ns'] <= s['join_begin_ns'])
        else:
            check('replay:no-join:' + label + '/' + str(j), s['join_begin_ns'] == s['join_end_ns'] == 0)
    ms = {m: describe([x[m] for x in by_block]) for m in 'OCSP'}
    pair = {'C_minus_O_ms': [x['C'] - x['O'] for x in by_block], 'S_minus_P_ms': [x['S'] - x['P'] for x in by_block], 'O_minus_P_ms': [x['O'] - x['P'] for x in by_block], 'O_over_P': [x['O'] / x['P'] for x in by_block]}
    compare_report(ms, rpa['cells'][i]['ms'], label + '/timing')
    for key, v in pair.items():
        compare_report(describe(v), rpa['cells'][i]['paired'][key]['summary'], label + '/' + key)
        check('reported:' + label + '/' + key + '/raw', np.allclose(v, rpa['cells'][i]['paired'][key]['raw'], rtol=2e-11, atol=1e-11))
    replay_out.append({'cell': i, 'layer': cell['layer'], 'H': cell['matched_H'], 'gpu_slots': cell['gpu_slots'], 'cold_slots_byte_equal': cold, 'correctness': got, 'reference': ref_out, 'timed_ms': ms, 'paired': {k: describe(v) for k, v in pair.items()}, 'first_use_ms': {s['mode']: (s['output_ready_ns'] - s['input_ready_ns']) / 1e6 for s in first}})
result['operator'] = {'cells': replay_out, 'GPU_ready_measured': False, 'source_GPU_call_count_not_hardware_trace': True, 'runtime_native_bytes_readback_verified_receipt': rp['bank_native_bytes_verified'], 'all_saved_operator_arrays_finite': True}

# Full-model blocks and 5 x 65 x 129280 logits, never sampling outputs here.
ian = js(ROOT / 'INTEGRATED_ANALYSIS.json')
names = {'F-before': '20261117T054354Z-F-before-U1024-fc9df30e', 'CSP': '20261117T054816Z-CSP-U1024-2ff29532', 'F-after': '20261117T055916Z-F-after-U1024-cc94b0a3'}
case = next(x for x in js(ROOT / 'integrated/CASES.json')['cases'] if x['case_id'] == 'text_cache_budget_ja')
tokens = case['expected_token_ids']; prefix = case['prefix_tokens']
check('integrated:text-hash', hashlib.sha256(case['text_utf8'].encode()).hexdigest() == case['sha256_utf8'])
check('integrated:train-only-case', case['split'] == 'train' and case['capture_calls'] == 64 and prefix == 272)
times = collections.defaultdict(list); block_stats = []; quality_arrays = {}; quality_rows = {}; histograms = collections.defaultdict(list)
common_run = runs[names['F-before']]
for arm, name in names.items():
    run = runs[name]; d = ROOT / 'integrated/runs' / name
    for k in ['cparams', 'placement', 'native_expert_inventory', 'cases', 'providers', 'tokenizer']:
        check('integrated:common:' + arm + '/' + k, run[k] == common_run[k])
    rc = next(x for x in run['cases'] if x['case_id'] == case['case_id'])
    check('integrated:tokens:' + arm, rc['full_tokens'] == tokens and rc['text_sha256'] == case['sha256_utf8'])
    bs = run['blocks']
    if arm == 'CSP':
        check('integrated:CSP-quality-order', [b['mode'] for b in bs if b['quality']] == list('CSP'))
        check('integrated:CSP-timing-order', ''.join(b['mode'] for b in bs if not b['quality']) == 'CSPPSCSPCCPSPCSSCP')
    else:
        check('integrated:F-blocks:' + arm, [(b['quality'], b['repetition']) for b in bs] == [(True, 0), (False, 0), (False, 1), (False, 2)])
    previous_end = 0
    for bi, b in enumerate(bs):
        label = arm if arm != 'CSP' else b['mode']; detail = f'{arm}/block{bi}'
        check('integrated:inputs:' + detail, b['prefix_tokens'] == prefix and b['input_tokens'] == tokens[prefix:prefix+64] and b['next_targets'] == tokens[prefix:prefix+65])
        check('integrated:statuses:' + detail, b['status'] == [0] * 64)
        a, z = np.asarray(b['start_ns'], dtype=np.int64), np.asarray(b['output_ready_ns'], dtype=np.int64)
        check('integrated:boundary-count:' + detail, a.size == z.size == 64)
        check('integrated:boundary-order:' + detail, previous_end <= b['prefill_begin_ns'] < b['prefill_ready_ns'] <= int(a[0]) and np.all(z > a) and np.all(a[1:] >= z[:-1]))
        previous_end = int(z[-1]); dt = (z - a) / 1e6
        stat = {'arm': arm, 'mode': b['mode'], 'quality': b['quality'], 'repetition': b['repetition'], 'mean_ms': float(dt.mean()), 'sum_ms': float(dt.sum()), 'prefill_ms': (b['prefill_ready_ns'] - b['prefill_begin_ns']) / 1e6, 'start_ns': int(a[0]), 'output_ready_ns': int(z[-1]), 'inter_call_gap_ms': describe((a[1:] - z[:-1]) / 1e6)}
        if not b['quality']:
            times[label].append(dt)
        if arm == 'CSP':
            h = np.asarray(b['after']['H']) - np.asarray(b['before']['H'])
            calls = np.asarray(b['after']['calls']) - np.asarray(b['before']['calls'])
            check('integrated:H-shape:' + detail, h.shape == (40, 7) and np.all(h >= 0))
            check('integrated:H-calls:' + detail, calls.tolist() == [64 if l in active else 0 for l in range(40)] and np.array_equal(h.sum(axis=1), calls))
            check('integrated:23-builds:' + detail, b['after']['builds'] - b['before']['builds'] == 23)
            hh = h.sum(axis=0).tolist()
            histograms[b['mode']].append({'quality': b['quality'], 'repetition': b['repetition'], 'per_layer': h.tolist(), 'total': hh, 'gpu_slots': sum(j * n for j, n in enumerate(hh))})
            stat['H'] = hh
        if b['quality']:
            logits = f32(d / b['file'], 65 * 129280).reshape(65, 129280)
            quality_arrays[label] = logits
            rows = []
            for j, target in enumerate(b['next_targets']):
                row = logits[j]; v = row.astype(np.float64); vmax = float(v.max())
                nll = vmax + math.log(float(np.exp(v - vmax).sum())) - float(v[target])
                top = int(np.argmax(row)); top2 = np.partition(v, -2)[-2:]
                stats = {'row': j, 'next_target': target, 'sha256': hashlib.sha256(row.tobytes()).hexdigest(), 'top1': top, 'margin': float(top2[-1] - top2[-2]), 'target_logit': float(v[target]), 'NLL': nll}
                compare_report(stats, ian['quality'][label][j], 'quality/' + label + '/' + str(j)); rows.append(stats)
            quality_rows[label] = rows
        block_stats.append(stat)

timing_summary = {}
for label, blocks in times.items():
    v = np.concatenate(blocks)
    row = {'total_tokens': v.size, 'total_ms': float(v.sum()), 'tokens_per_second': 1000 * v.size / float(v.sum()), 'token_ms': describe(v), 'block_mean_ms': describe([x.mean() for x in blocks])}
    compare_report(row, ian['timing_summary'][label], 'timing/' + label)
    timing_summary[label] = row
check('integrated:1536-timing-calls', sum(x['total_tokens'] for x in timing_summary.values()) == 1536)
comparisons = {}
for left, right in [('F-after', 'F-before'), ('C', 'F-before'), ('S', 'F-before'), ('P', 'F-before'), ('S', 'P')]:
    a, b = quality_arrays[left], quality_arrays[right]
    changed = [i for i in range(65) if not equality(a[i], b[i])]
    diffnll = [quality_rows[left][i]['NLL'] - quality_rows[right][i]['NLL'] for i in range(65)]
    m = metric(a, b)
    comp = {'byte_equal': m['byte_equal'], 'rows_changed': changed, 'first_changed_row': changed[0] if changed else None, 'top1_changed_rows': [i for i in range(65) if quality_rows[left][i]['top1'] != quality_rows[right][i]['top1']], 'NMSE': m['nmse'], 'max_abs': m['max_abs'], 'mean_reference_token_NLL_delta': float(np.mean(diffnll)), 'row_NLL_delta': diffnll}
    compare_report(comp, ian['quality_comparisons'][left + '_vs_' + right], left + '_vs_' + right)
    comparisons[left + '_vs_' + right] = comp
check('integrated:F-before-F-after-C-byte', comparisons['F-after_vs_F-before']['byte_equal'] and comparisons['C_vs_F-before']['byte_equal'])
check('integrated:S-P-byte', comparisons['S_vs_P']['byte_equal'])
prior_five = np.fromfile(PRIOR / 'text_cache_budget_ja.f32', dtype='<f4').reshape(5, 129280)
check('integrated:F-prior-five-byte', equality(quality_arrays['F-before'][:5], prior_five))
check('integrated:H-stable-all-SP', all(x['per_layer'] == histograms['S'][0]['per_layer'] for m in 'SP' for x in histograms[m]))
training_hist = [0] * 7
training_per_layer = [[0] * 7 for _ in range(40)]
for r in base_records:
    if r['decode_call'] < 64 and r['layer'] in active:
        h = sum(x in selected[r['layer']] for x in r['ids'])
        training_hist[h] += 1; training_per_layer[r['layer']][h] += 1
train_slots = sum(i * n for i, n in enumerate(training_hist))
f = np.concatenate(times['F-before'] + times['F-after'])
fbracket = {'mean_ms': float(f.mean()), 'tokens_per_second': 1000 / float(f.mean()), 'total_tokens': f.size}
compare_report(fbracket, ian['F_bracket'], 'F_bracket')
result['integrated'] = {'timing_summary': timing_summary, 'F_bracket': fbracket, 'speed_vs_F': {x: float(f.mean()) / timing_summary[x]['token_ms']['mean'] for x in 'CSP'}, 'blocks': block_stats, 'quality_comparisons': comparisons, 'quality_rows': quality_rows, 'total_quality_f32_values': sum(x.size for x in quality_arrays.values()), 'training_active_layer_H': training_hist, 'training_active_gpu_slots': train_slots, 'integrated_active_layer_H': histograms['S'][0]['total'], 'integrated_active_gpu_slots': histograms['S'][0]['gpu_slots'], 'H_per_layer_changes': [{'layer': l, 'training': training_per_layer[l], 'integrated': histograms['S'][0]['per_layer'][l]} for l in active if training_per_layer[l] != histograms['S'][0]['per_layer'][l]], 'histograms': dict(histograms), 'quality_scope': '65 conditional rows of one already-used JA training trajectory; no free generation or accuracy benchmark', 'performance_scope': '1536 timed decode calls, 24 blocks, 3 processes; 5 additional quality blocks excluded from timing'}

restore = js(ROOT / 'RESTORED_RUNTIME.json')
check('restore:API', restore['normal_API_restored'] and restore['API_health'] == 'ok')
check('restore:original-runtime', restore['normal_runtime_backing_files_verified'] and not restore['diagnostic_libraries_loaded'])
check('restore:services', all(x == 'active' for x in restore['services'].values()))
check('restore:alerts', not restore['monitor_errors'] and not restore['active_alerts'])
check('restore:config-pin', all(js(ROOT / 'integrated/runs' / n / 'LAUNCH_MANIFEST.json')['baseline_config_sha256'] == restore['config_sha256'] for n in names.values()))
state = js(ROOT / 'STATE.json')
check('restore:not-adopted', state['integrated_candidate_adopted'] is False and state['production_changed'] is False)
result['restore_saved_receipt'] = restore
result['limits'] = [
    'ELF/model bytes are not published here: runtime/library statements are source/provenance consistency checks, not independent ELF execution or hash reads from the live target.',
    'Native bank upload readback is an original-run receipt; this audit cannot re-read target GPU memory.',
    'Only baseline operator outputs are stored; later timed output equality is checked by original source and successful run receipt, not independent access to all output tensors.',
    'Saved reference tensors allow an independent metric recomputation; reference dequantization itself was not rerun because the model is unavailable.',
    'Matching route histograms does not prove identical token-by-token routing. Main CSP collector remained closed.',
    'The sample is one already-used JA training trajectory; not independent prompts, unseen quality tasks, long-context stress or HTTP/SSE/WebUI latency.',
    'F is the integrated diagnostic engine with the hot path disabled; it is not a new production frozen-binary measurement.',
    'Capture timing is invalid for speed conclusions; final capture matches 333 earlier route records and 5 logits rows, but call5/8 callback-free intermediate identity remains unproven.',
    'CUDA memory and NVML queries are sequential; reported phases and health tails do not establish continuous peak usage or arbitrary future workload admission.',
]
result['source_files'] = [source_info(ROOT / x) for x in ['README.txt', 'CONSULT_PRO_JA.txt', 'RESULTS_JA.txt', 'INTEGRATED_ANALYSIS.json', 'REPLAY_ANALYSIS.json', 'CAPTURE_ANALYSIS.json', 'integrated/driver.cpp', 'integrated/hot-backend.cpp', 'integrated/ENGINE_BUILD.json', 'replay/operator_replay.h', 'capture/driver.cpp', 'RESTORED_RUNTIME.json']]
result['checks_by_group'] = dict(checks)
result['total_checks'] = sum(checks.values())
result['failures'] = failures
result['passed'] = not failures
result['auditor_sha256'] = sha(Path(__file__))
A.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x)) + '\n')
print(json.dumps({'passed': result['passed'], 'checks': result['total_checks'], 'failures': failures, 'output': str(A.output), 'timing_tokens_per_second': {k: v['tokens_per_second'] for k, v in timing_summary.items()}, 'F_bracket': fbracket, 'P_vs_F': comparisons['P_vs_F-before'], 'training_H': training_hist, 'integrated_H': histograms['S'][0]['total']}, ensure_ascii=False, indent=2))
sys.exit(0 if result['passed'] else 1)
