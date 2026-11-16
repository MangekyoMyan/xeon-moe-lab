#!/usr/bin/env python3
"""Read saved evidence and verify a proposed sampling assignment; execute no workload."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

COMMIT = 'ff6168094858141aeeebf45abc5c309882fe5216'
N = Path('archive/review-handoffs-not-for-readers/2026-11-17-mixed-operator')
A = Path('archive/review-handoffs-not-for-readers/2026-11-16-abs-operator')
I = N/'incoming/xeon_moe_abs_next_2026-11-16/next_experiment'


def record(repo, path):
    payload = (repo/path).read_bytes()
    return {'repo_relative_path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def logical(block, arm):
    reverse = ((block % 8) + (block // 8)) % 2
    middle = arm in (1, 2)
    return 'S' if middle != bool(reverse) else 'O'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    repo = args.repo.resolve()
    paths = [N/name for name in ('CONSULT_PRO_JA.txt', 'RESULTS_JA.txt', 'README.txt', 'POLICY_AND_PLAN_JA.txt',
             'evaluate_screen.py', 'run_mixed_suite.py', 'build_mixed.py', 'build/BUILD.json',
             'SCREEN_DECISION.json', 'MIXED_ANALYSIS.json')]
    paths += [I/name for name in ('mixed_probe.cpp', 'run_one_mixed.py', 'analyze_mixed_run.py')]
    paths += [A/name for name in ('ggml-cpu.c', 'iq-abs.h', 'guard.py')]
    processes = []
    ids = {}
    c0 = {}
    for index in range(8):
        folder = N/'mixed-run'/('process-'+str(index))
        guard_path = N/'outer-guard'/('iq2-mixed-20261117-p'+str(index))/'STATUS.json'
        paths += [folder/'STATUS.json', folder/'raw.jsonl', folder/'maps.txt', guard_path]
        status = json.loads((repo/folder/'STATUS.json').read_text())
        raw = [json.loads(line) for line in (repo/folder/'raw.jsonl').read_text().splitlines()]
        guard = json.loads((repo/guard_path).read_text())
        meta = raw[0]
        chunks = [r for r in raw if r.get('kind') == 'mixed_chunk']
        c0[index] = [r for r in chunks if r['case'] == 'C0']
        assert len(c0[index]) == 64 and len(chunks) == 192
        assert status['passed'] and status['state'] == 'completed' and status['exit_code'] == 0 and status['error'] is None
        assert guard['exit_code'] == 0 and guard['error'] is None
        assert status['library_label'] == 'FEEFEFFE'[index]
        for key in ('base_library_sha256', 'harness_sha256', 'inputs_sha256', 'plan_sha256', 'launcher_sha256'):
            ids.setdefault(key, set()).add(status['identity'][key])
        samples = [guard['initial'], *guard['samples']]
        host_keys = sorted(set().union(*(row.get('host_counters', {}).keys() for row in samples)))
        maps = (repo/folder/'maps.txt').read_text().splitlines()
        libgomp = sorted({line.split()[-1] for line in maps if '/libgomp' in line})
        processes.append({
            'process_index': index, 'library_label': status['library_label'],
            'case_order': meta['case_order'], 'launcher_elapsed_ns': status['ended_monotonic_ns']-status['started_monotonic_ns'],
            'c0_chunk_count': len(c0[index]),
            'c0_first_begin_ns': min(r['begin_ns'] for r in c0[index]),
            'c0_last_end_ns': max(r['end_ns'] for r in c0[index]),
            'c0_totals_ns': {mode: sum(r['elapsed_ns'] for r in c0[index] if r['logical_mode'] == mode) for mode in ('O','S')},
            'chunk_fields': sorted(chunks[0]),
            'guard_sample_count_including_initial': len(samples),
            'guard_sensor_unix_values': sorted({row['unix'] for row in samples}),
            'guard_started_unix': guard['started_unix'],
            'guard_finished_unix': guard['finished_unix'],
            'guard_api_cores_max': max(row.get('api_cores', 0) for row in samples),
            'guard_host_counters_unchanged': all(row['host_counters'] == samples[0]['host_counters'] for row in samples),
            'host_counter_count': len(host_keys),
            'host_counter_categories': sorted({'thermal_throttle' if '/thermal_throttle/' in k else 'edac' if '/edac/' in k else 'other' for k in host_keys}),
            'saved_maps_include_weight_shard': any('.gguf' in line for line in maps),
            'saved_libgomp_paths': libgomp,
            'controlled_env': status['env']['controlled_values'],
        })
    assert all(len(values) == 1 for values in ids.values())
    pairs = []
    for frozen, overlay in ((0,1),(3,2),(5,4),(6,7)):
        f_total = sum(r['elapsed_ns'] for r in c0[frozen])
        assert f_total < 2**53  # Integer and half-ns values below remain exactly representable.
        pair = {'frozen_process': frozen, 'overlay_process': overlay, 'frozen_all_arms_total_ns': f_total}
        for mode in ('O','S'):
            e_total = sum(r['elapsed_ns'] for r in c0[overlay] if r['logical_mode'] == mode)
            pair['overlay_'+mode+'_total_ns'] = e_total
            pair['twice_overlay_'+mode+'_minus_frozen_ns'] = 2*e_total-f_total
            pair['overlay_'+mode+'_minus_half_frozen_ns'] = e_total - f_total/2
        pairs.append(pair)
    assignment = []
    for block in range(16):
        counts = Counter()
        for arm in range(4):
            mode = logical(block, arm)
            enabled = (arm + block % 8) & 1
            assignment.append({'block': block, 'route_group': block % 8, 'arm': arm,
                               'logical_mode': mode, 'diagnostic_enabled': bool(enabled)})
            counts[(mode, enabled)] += 1
        assert counts == Counter({('O',0):1, ('O',1):1, ('S',0):1, ('S',1):1})
    by_group = {}
    for group in range(8):
        counts = Counter((row['logical_mode'], int(row['diagnostic_enabled'])) for row in assignment if row['route_group'] == group)
        assert set(counts.values()) == {2}
        by_group[str(group)] = {mode+'/'+str(enabled): counts[(mode,enabled)] for mode in ('O','S') for enabled in (0,1)}
    result = {
        'schema': 1, 'pinned_commit': COMMIT,
        'scope': 'Offline saved-file reading and integer protocol assignment checks only; no new timer/counter collection, model, CPU library, API, GPU, or performance run.',
        'source_manifest': [record(repo,p) for p in paths],
        'shared_identity': {key: next(iter(values)) for key,values in ids.items()},
        'processes': processes, 'c0_pair_total_context': pairs,
        'missing_from_current_chunk_schema': ['process_cpu_ns','rusage_self','rusage_thread','context_switches','page_faults','perf','cgroup_cpu_stat','clock_pair_realtime'],
        'proposed_half_chunk_assignment': {'executed': False, 'formula': '(arm + block % 8) & 1',
            'all_wall_chunks_retained': 64, 'counter_enabled_chunks': 32, 'counter_disabled_chunks': 32,
            'within_each_block_each_logical_mode_on_off_counts': [1,1],
            'within_each_route_group_each_logical_mode_on_off_counts': by_group, 'rows': assignment},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'output':str(args.output),'manifest_files':len(paths),'processes':len(processes),'assignment_checked':True}))


if __name__ == '__main__':
    main()
