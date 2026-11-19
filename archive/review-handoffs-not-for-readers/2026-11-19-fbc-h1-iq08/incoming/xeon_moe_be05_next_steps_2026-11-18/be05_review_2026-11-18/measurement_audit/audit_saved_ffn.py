#!/usr/bin/env python3
"""Independent read-only audit of the be05 actual-Xeon saved FFN trial.

The engine, provided runner and analysis scripts are never imported or executed.
Usage: audit_saved_ffn.py --source ../source --capture capture_reference --out result.json
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median, mean
import struct

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--capture', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root, caproot = args.source.resolve(), args.capture.resolve()
q = root / 'q8-avx1'
read_receipts = {}


def need(test, why):
    if not test:
        raise RuntimeError(why)


def raw(path):
    b = path.read_bytes()
    for label, directory in [('source', root), ('capture', caproot)]:
        if path.is_relative_to(directory):
            name = label + '/' + str(path.relative_to(directory))
            break
    else:
        name = str(path)
    read_receipts[name] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
    return b


def js(path):
    return json.loads(raw(path))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fnv(data):
    n = 14695981039346656037
    for x in data:
        n = ((n ^ x) * 1099511628211) & ((1 << 64) - 1)
    return n


design = js(q / 'FFN_DESIGN.json')
published = js(q / 'FFN_ANALYSIS.json')
state = js(q / 'STATE-v2.json')
build = js(q / 'BUILD.json')
ffn_build = js(q / 'FFN_BUILD.json')
hardware = js(root / 'HARDWARE.json')
capture = js(caproot / 'CAPTURE.json')
capmanifest = js(caproot / 'RESULTS_SHA256.json')
need(sha(raw(caproot / 'RESULTS_SHA256.json')) == design['common']['capture_manifest_sha256'], 'capture manifest identity')
need(sha(raw(caproot / 'CAPTURE.json')) == capmanifest['CAPTURE.json'], 'capture JSON identity')
need(sha(raw(q / 'BUILD.json')) == design['CPU_build_sha256'], 'CPU build receipt')
need(sha(raw(q / 'FFN_BUILD.json')) == design['FFN_build_sha256'], 'FFN build receipt')
need(sha(raw(q / 'real_ffn.cpp')) == ffn_build['source_sha256'], 'FFN source identity')
need(build['unchanged_object_byte_equal'] and build['control_CPU_byte_equal_live'], 'recorded current control')
need(build['original_CPU_sha256'] == '33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3', 'current CPU identity')
need(state['completed'] and state['normal_API_restored'] and not state['restore_required'], 'recorded completion/restoration')

selected_capture = {}
for cell in [1, 5, 6]:
    c = capture['cells'][cell]
    need(c['cell'] == cell and c['x_ne'] == [5120, 1, 1, 1], 'T1 captured input')
    need(c['ids_ne'] == [6, 1, 1, 1] and len(c['ids']) == 6, 'captured top6')
    blobs = {k: raw(caproot / c[k]) for k in ['x_file', 'ids_file', 'down_file', 'routing_weights_file']}
    for k, data in blobs.items():
        need(sha(data) == capmanifest[c[k]], 'captured file hash')
    need(list(struct.unpack('<6i', blobs['ids_file'])) == c['ids'], 'ordered top6 bytes')
    need(len(blobs['x_file']) == 5120 * 4 and len(blobs['down_file']) == 5120 * 6 * 4, 'capture sizes')
    need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', blobs['down_file'])), 'finite capture outputs')
    selected_capture[cell] = (c, blobs)

topology = hardware['CPU0_23_topology']
need(len({(x['socket'], x['core']) for x in topology if x['cpu'] < 12}) == 12, '12 CPU IDs are distinct physical cores')
need(len({(x['socket'], x['core']) for x in topology}) == 12, '24 CPU IDs cover 12 cores')

timing_order = ['control', 'candidate', 'candidate', 'control', 'candidate', 'control', 'control', 'candidate']
need([x['mode'] for x in design['stages'] if x['stage'] == 'timing'] == timing_order, 'ABBA BAAB design')
need(len(design['stages']) == 10, 'ten stages')
stages, blocks_by_stage = {}, {}
saved_count = timed_samples = timed_calls = 0
manifest_entries_checked = 0
process_ids, process_periods = set(), []
baseline_weights = {}
sample_table = []
for stage in design['stages']:
    name, mode = stage['name'], stage['mode']
    folder = q / 'runs-v2' / name
    run = js(folder / 'RUN.json')
    spec = js(folder / 'SPEC.json')
    launch = js(folder / 'LAUNCH.json')
    health = js(folder / 'guard-health.json')
    manifest = js(folder / 'RESULTS_SHA256.json')
    need(sha(raw(folder / 'RESULTS_SHA256.json')) == state['runs'][name]['manifest_sha256'], 'run manifest receipt')
    for file_name, digest in manifest.items():
        need(Path(file_name).name == file_name, 'flat run manifest')
        if file_name == 'health.json':
            data = raw(folder / 'guard-health.json')
        elif file_name == 'server-config.json':
            data = (json.dumps({'command': launch['command'], 'environment': launch['environment']}, indent=2) + '\n').encode()
        else:
            data = raw(folder / file_name)
        need(sha(data) == digest, 'run file SHA: ' + name + '/' + file_name)
        manifest_entries_checked += 1
    need(run['passed'] and run['spec'] == spec, 'successful exact spec')
    need(spec == dict(design['common'], **stage, CPU=spec['CPU']), 'stage/common spec consistency')
    need(launch['design_sha256'] == sha(raw(q / 'FFN_DESIGN.json')), 'design pin')
    need(launch['spec_sha256'] == sha(raw(folder / 'SPEC.json')), 'spec pin')
    need(launch['CPU_sha256'] == build['variants'][mode]['CPU_sha256'], 'CPU variant pin')
    need(launch['binary_sha256'] == ffn_build['binary_sha256'], 'FFN probe binary pin')
    need(launch['base_sha256'] == '2e13b3701059c1529b97b185bdc659d3dc4730e0763ed3bb8eb9cdb9563cc558', 'common base pin')
    need(launch['runner_sha256'] == sha(raw(root / 'run_ffn.py')), 'runner source pin')
    need(health['status'] == 'exited' and health['child_returncode'] == 0, 'successful child exit')
    need(health['pid'] not in process_ids, 'fresh process PID')
    process_ids.add(health['pid'])
    process_periods.append({'name': name, 'start_unix': health['started_unix'], 'end_unix': health['finished_unix']})
    expected_cpu = spec['CPU']
    for key in ['CPU', 'cache', 'from_float']:
        need(run['providers'][key] == expected_cpu, 'provider ' + key)
    need(run['providers']['base'] == spec['base'], 'base provider')
    command = launch['command']
    need('numactl' in command and '--physcpubind=0-23' in command and '--interleave=0,1' in command, 'NUMA/affinity command')
    need(any(x.endswith('without_thp.py') for x in command), 'THP wrapper')
    env = launch['environment']
    need(env['LD_PRELOAD'] == '' and env['OMP_DYNAMIC'] == 'FALSE' and env['GGML_CPU_IQ_ROW_LOCAL'] == '0', 'recorded runtime environment')
    keys = [(b['cell'], b['threads']) for b in run['blocks']]
    need(keys == [(c, t) for c in stage['cells'] for t in stage['threads']], 'within-process condition order')
    stage_blocks = {}
    previous_end = None
    for b in run['blocks']:
        cell, threads = b['cell'], b['threads']
        c, cb = selected_capture[cell]
        need(b['ids'] == c['ids'] and b['layer'] == c['layer'], 'captured route/layer')
        need(b['x_FNV64'] == fnv(cb['x_file']), 'captured x FNV')
        need(b['output_matches_capture'], 'recorded output equality')
        need(sorted(b['worker_receipt_after_timing']['CPU']) == list(range(threads)), 'post-timing CPU receipt')
        need(len(set(b['worker_receipt_after_timing']['TID'])) == threads, 'post-timing unique worker TID')
        need(len(b['weights']) == 3, 'three projections')
        for w, cp in zip(b['weights'], c['projections']):
            need(w['name'] == cp['name'] and w['type'] == cp['type'] and w['ne'] == cp['ne'], 'projection identity/shape')
            compact = [e for e, enabled in zip(c['ids'], cp['compact_mask']) if enabled]
            need(w['compact_experts'] == compact, 'actual selected compact list')
            planned = spec['cpu_plan'].get(w['name'], [])
            need([e in planned for e in c['ids']] == cp['compact_mask'], 'projection plan membership')
            need([x['expert'] for x in w['selected_native']] == c['ids'], 'ordered native records')
            need(all(x['bytes'] == cp['nb'][2] for x in w['selected_native']), 'native expert byte size')
        if cell not in baseline_weights:
            baseline_weights[cell] = b['weights']
        need(b['weights'] == baseline_weights[cell], 'unchanged selected weight records in every process')
        output = raw(folder / b['output_file'])
        need(output == cb['down_file'], 'saved output bytes equal captured output')
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', output)), 'finite saved output')
        saved_count += 1
        need(len(b['samples']) == stage['samples'], 'sample count')
        values = []
        for number, s in enumerate(b['samples']):
            need(s['sample'] == number and s['calls'] == stage['loops'], 'sample index/call count')
            need(s['end_ns'] > s['begin_ns'], 'positive interval')
            if previous_end is not None:
                need(s['begin_ns'] >= previous_end, 'nonoverlapping ordered samples')
            previous_end = s['end_ns']
            need(s['output_FNV64'] == fnv(output), 'sample-end FNV matches output')
            value = (s['end_ns'] - s['begin_ns']) / s['calls'] / 1e6
            values.append(value)
            if stage['stage'] == 'timing':
                timed_calls += s['calls']; timed_samples += 1
                sample_table.append({'process': name, 'mode': mode, 'cell': cell, 'threads': threads,
                    'condition_index': keys.index((cell, threads)), 'first_fixture': cell == stage['cells'][0],
                    'sample': number, 'begin_ns': s['begin_ns'], 'end_ns': s['end_ns'],
                    'graph_calls': s['calls'], 'ms_per_graph': value})
        stage_blocks[cell, threads] = {'values': values, 'median_ms': median(values),
                                     'first_fixture': cell == stage['cells'][0]}
    stages[name] = stage
    blocks_by_stage[name] = stage_blocks

need(saved_count == 60 and timed_calls == 1920 and timed_samples == 240, 'complete output/timing coverage')
for a, b in zip(process_periods, process_periods[1:]):
    need(a['end_unix'] <= b['start_unix'], 'processes did not overlap')
rows = []
pairs = [(0, 1), (3, 2), (5, 4), (6, 7)]
for cell in [1, 5, 6]:
    for threads in [12, 24]:
        published_row = next(r for r in published['rows'] if r['cell'] == cell and r['threads'] == threads)
        medians, within = {}, []
        for i, mode in enumerate(timing_order):
            block = blocks_by_stage[f'timing-{i}-{mode}'][cell, threads]
            medians[i] = block['median_ms']
            need(medians[i] == published_row['processes'][i]['median_ms'], 'independent process median')
            within.append({'process': i, 'mode': mode, 'first_fixture': block['first_fixture'],
                           'median_ms': block['median_ms'], 'min_ms': min(block['values']),
                           'max_ms': max(block['values']), 'max_over_min': max(block['values']) / min(block['values'])})
        ratios = [medians[p] / medians[c] for c, p in pairs]
        result = {'cell': cell, 'threads': threads, 'ratios': ratios, 'ratio_median': median(ratios),
                  'wins': sum(x < 1 for x in ratios), 'range': [min(ratios), max(ratios)],
                  'saved_ms': [medians[c] - medians[p] for c, p in pairs], 'process_sample_ranges': within}
        need(result['ratio_median'] == published_row['median_time_ratio'], 'independent paired median')
        need(result['wins'] == published_row['pairs_faster'], 'independent wins')
        need(result['range'] == [published_row['min_ratio'], published_row['max_ratio']], 'independent ratio range')
        rows.append(result)

first24 = [p for row in rows if row['threads'] == 24 for p in row['process_sample_ranges'] if p['first_fixture']]
later24 = [p for row in rows if row['threads'] == 24 for p in row['process_sample_ranges'] if not p['first_fixture']]
result = {'status': 'pass', 'scope': 'Independent saved evidence audit only; no engine/build/provided script executed',
          'successful_processes': len(stages), 'unique_child_pids': len(process_ids),
          'run_manifest_entries_checked': manifest_entries_checked, 'saved_output_files_compared': saved_count,
          'saved_output_F32_values': saved_count * 5120 * 6, 'timed_samples': timed_samples,
          'timed_graph_calls': timed_calls, 'saved_output_vs_capture_byte_equal': True,
          'all_six_paired_statistics_match_published': True, 'rows': rows,
          'order_association': {'method': 'Post-hoc descriptive grouping; no samples excluded, no causal conclusion',
              'first_fixture_threads24_count': len(first24),
              'first_fixture_threads24_sample_max_min_range': [min(p['max_over_min'] for p in first24), max(p['max_over_min'] for p in first24)],
              'first_fixture_threads24_sample_max_min_median': median(p['max_over_min'] for p in first24),
              'later_fixture_threads24_count': len(later24),
              'later_fixture_threads24_sample_max_min_range': [min(p['max_over_min'] for p in later24), max(p['max_over_min'] for p in later24)],
              'later_fixture_threads24_sample_max_min_median': median(p['max_over_min'] for p in later24)},
          'process_periods': process_periods, 'input_file_receipts': read_receipts,
          'limits': ['Provider/ELF/GGUF claims are recorded identities; model shards and runtime ELF are not supplied here',
                     'The worker receipt is a separate OpenMP region after timing, not a per-call affinity trace',
                     'One output per condition/process is saved; timed output checks are per sample end, not every graph call',
                     'The three captured cells come from one saved training trajectory',
                     'NUMA policy is recorded; file-backed page cache residency and full-model cache pressure are not reconstructed']}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
with args.out.with_name('INDEPENDENT_TIMING_SAMPLES.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(sample_table[0]))
    writer.writeheader(); writer.writerows(sample_table)
print(json.dumps({k: result[k] for k in ['status', 'successful_processes', 'saved_output_files_compared', 'timed_samples', 'timed_graph_calls', 'order_association']}, ensure_ascii=False))
