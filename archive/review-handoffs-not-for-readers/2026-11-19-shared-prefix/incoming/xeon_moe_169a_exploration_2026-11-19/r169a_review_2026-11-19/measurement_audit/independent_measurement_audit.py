#!/usr/bin/env python3
"""Read-only audit of saved raw observations, without provider verify/analyze code.

Only JSON, text plans and F32/I32 files are read. No model, DSO, command from a
receipt, original verifier, API, compilation or inference is executed. Output is
written exclusively under --out. Binary comparisons are literal byte comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import struct
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'source')
parser.add_argument('--capture', type=Path, default=Path(__file__).resolve().parents[1] / 'capture_reference')
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
src, cap, out = args.source.resolve(), args.capture.resolve(), args.out.resolve()
assert not out.is_relative_to(src) and not out.is_relative_to(cap)
out.mkdir(parents=True, exist_ok=True)

checks = []
inputs = {}


def require(cond, text):
    checks.append({'passed': bool(cond), 'check': text})
    if not cond:
        raise AssertionError(text)


def digest(path):
    path = Path(path)
    if str(path) not in inputs:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(1048576), b''):
                h.update(b)
        inputs[str(path)] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return inputs[str(path)]['sha256']


def read_json(path):
    digest(path)
    return json.loads(path.read_bytes())


def bytes_of(path):
    digest(path)
    return path.read_bytes()


def binary_record(path, expected_floats):
    digest(path)
    require(path.stat().st_size == expected_floats * 4, f'{path.name}: expected F32 length')
    data = np.memmap(path, dtype='<f4', mode='r')
    # Chunked scan, also for the six 33.6 MB logit files.
    finite = all(bool(np.isfinite(data[i:i + 262144]).all()) for i in range(0, len(data), 262144))
    require(finite, f'{path.name}: all saved F32 finite')
    del data
    return {'path': str(path.relative_to(src)) if path.is_relative_to(src) else str(path),
            'sha256': digest(path), 'bytes': path.stat().st_size,
            'f32_count': expected_floats, 'all_finite': finite}


def stream_equal(a, b):
    digest(a); digest(b)
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open('rb') as fa, b.open('rb') as fb:
        while True:
            ba, bb = fa.read(1048576), fb.read(1048576)
            if ba != bb:
                return False
            if not ba:
                return True


def fnv64(data):
    h = 14695981039346656037
    for v in data:
        h = ((h ^ v) * 1099511628211) & ((1 << 64) - 1)
    return h


def describe_ns(values):
    return {'n': len(values), 'mean_ms': st.mean(values) / 1e6,
            'median_ms': st.median(values) / 1e6,
            'min_ms': min(values) / 1e6, 'max_ms': max(values) / 1e6,
            'positive': sum(x > 0 for x in values), 'negative': sum(x < 0 for x in values),
            'zero': sum(x == 0 for x in values)}


# Process facts are saved records, not a claim about the current machine state.
processes = []
for family in ('fbc', 'h1', 'cpu08'):
    for d in sorted((src / family / 'runs').iterdir()):
        if not d.is_dir():
            continue
        g = read_json(d / 'guard-health.json')
        require(g['child_returncode'] == 0 and g['status'] == 'exited', f'{family}/{d.name}: saved exit status')
        processes.append({'family': family, 'run': d.name, 'pid': g['pid'], 'boot_id': g['boot_id'],
                          'started_unix': g['started_unix'], 'finished_unix': g['finished_unix']})
processes.sort(key=lambda x: x['started_unix'])
require(len(processes) == 23, '23 saved process records')
require(len({p['pid'] for p in processes}) == 23, '23 distinct recorded PIDs')
require(len({p['boot_id'] for p in processes}) == 1, 'same recorded boot ID')
require(all(a['finished_unix'] <= b['started_unix'] for a, b in zip(processes, processes[1:])),
        'all 23 saved guard process windows nonoverlapping')


# F/B/C: arithmetic means of all 64 T1 in each process/case, then half-pass pairs.
cases = {c['case_id']: c for c in read_json(src / 'fbc/CASES.json')['cases']}
fbc = {'timing_runs': [], 'quality_saved_outputs': [], 'endpoint_checks': [], 'cases': {}}
quality_paths = {}
for d in sorted((src / 'fbc/runs').iterdir()):
    r = read_json(d / 'RUN.json')
    require(r['passed'], f'{d.name}: saved caller pass')
    require(len(r['blocks']) == 2, f'{d.name}: two cases')
    rr = {'run': d.name, 'arm': r['arm'], 'case_order': [], 'blocks': []}
    for b in r['blocks']:
        name = b['case_id']; c = cases[name]
        rr['case_order'].append(name)
        require(b['prefix_ids'] == c['prefix_ids'], f'{d.name}/{name}: frozen prefix IDs')
        require(b['input_tokens'] == c['used_T1_ids'], f'{d.name}/{name}: frozen 64 teacher inputs')
        require(b['next_targets'] == c['next_targets_prefix_and64T1'], f'{d.name}/{name}: frozen 65 targets')
        require(b['positions'] == list(range(c['prefix_count'], c['prefix_count'] + 64)),
                f'{d.name}/{name}: 64 contiguous frozen positions')
        starts, ends = b['start_ns'], b['output_ready_ns']
        require(len(starts) == len(ends) == len(b['status']) == 64, f'{d.name}/{name}: 64 recorded T1')
        require(set(b['status']) == {0}, f'{d.name}/{name}: all recorded T1 status zero')
        require(all(isinstance(v, int) for v in starts + ends), f'{d.name}/{name}: integer ns')
        require(all(a < z for a, z in zip(starts, ends)), f'{d.name}/{name}: positive elapsed')
        require(all(a <= z for a, z in zip(ends, starts[1:])), f'{d.name}/{name}: sequential T1')
        duration = [z-a for a,z in zip(starts, ends)]
        block = {'case': name, 'position_in_process': len(rr['blocks']),
                 'durations_ns': duration, 'start_ns': starts, 'output_ready_ns': ends,
                 **describe_ns(duration)}
        rr['blocks'].append(block)
        if r['stage'] == 'quality':
            path = d / b['file']
            require(b['shape'] == [65, r['vocab_size']], f'{d.name}/{name}: quality rows/vocabulary')
            fbc['quality_saved_outputs'].append(binary_record(path, 65 * r['vocab_size']))
            quality_paths[(r['arm'], name)] = path
    if r['stage'] == 'timing':
        fbc['timing_runs'].append(rr)
fbc['timing_runs'].sort(key=lambda r: r['run'])
require([r['arm'] for r in fbc['timing_runs']] == list('FBCCBF'), 'FBC timing order F/B/C/C/B/F')
require([p['run'] for p in processes if p['family']=='fbc' and p['run'].startswith('timing-')] ==
        [r['run'] for r in fbc['timing_runs']], 'FBC run naming agrees with recorded chronological guard order')
ordered_cases = list(cases)
require(all(r['case_order']==(ordered_cases if i<3 else list(reversed(ordered_cases)))
            for i,r in enumerate(fbc['timing_runs'])), 'FBC case order reversed in second half-pass')
require(sum(b['n'] for r in fbc['timing_runs'] for b in r['blocks']) == 768, '768 timed T1 calls')
for name in cases:
    require(stream_equal(quality_paths[('F', name)], quality_paths[('B', name)]) and
            stream_equal(quality_paths[('F', name)], quality_paths[('C', name)]), f'{name}: quality F=B=C literal bytes')
    means = {}
    for i, r in enumerate(fbc['timing_runs']):
        b = next(b for b in r['blocks'] if b['case'] == name)
        means[i] = sum(b['durations_ns']) / 64
        last = src / 'fbc/runs' / r['run'] / ('last-' + name + '.f32')
        record = binary_record(last, 129280)
        with quality_paths[(r['arm'], name)].open('rb') as q:
            q.seek(64 * 129280 * 4); last_quality = q.read(129280 * 4)
        eq = bytes_of(last) == last_quality
        require(eq, f'{r["run"]}/{name}: timing endpoint equals quality row 64')
        fbc['endpoint_checks'].append({**record, 'equals_quality_last_row': eq})
    diffs = {'B_minus_F': [(means[1]-means[0])/1e6, (means[4]-means[5])/1e6],
             'C_minus_B': [(means[2]-means[1])/1e6, (means[3]-means[4])/1e6],
             'C_minus_F': [(means[2]-means[0])/1e6, (means[3]-means[5])/1e6]}
    fbc['cases'][name] = {'run_mean_ms': [means[i]/1e6 for i in range(6)],
                         'arm_mean_ms': {a: st.mean(means[i] for i,r in enumerate(fbc['timing_runs']) if r['arm']==a)/1e6 for a in 'FBC'},
                         'paired_differences_ms': diffs,
                         'mean_paired_differences_ms': {k: st.mean(v) for k,v in diffs.items()},
                         'F_last_minus_first_ms': (means[5]-means[0])/1e6,
                         'fresh_processes_per_arm_case': 2}


# Capture binary facts: independently obtained fixed-commit bytes, not model weights.
capture = read_json(cap / 'CAPTURE.json')
capture_cells = {c['cell']: c for c in capture['cells']}
capture_bytes = {}
for i, c in capture_cells.items():
    down = cap / c['down_file']
    binary_record(down, 6 * 5120)
    capture_bytes[i] = bytes_of(down)
    ids = list(struct.unpack('<6i', bytes_of(cap / c['ids_file'])))
    require(ids == c['ids'], f'capture cell{i}: actual I32 IDs equal metadata')


# H1: all saved mode outputs and all 12 paired blocks per cell are included.
h1 = {'runs': [], 'primary': {}, 'cold_slot_checks': 0, 'steady_observations': 0,
      'first_use_observations': 0, 'mode_output_files': 0, 'reference_files': 0,
      'partition_output_differences': {}}
output_bytes = {}
for d in sorted((src / 'h1/runs').iterdir()):
    replay = read_json(d / 'REPLAY.json')
    plan_name = replay['diagnostic_plan']
    plan_path = src / 'h1' / (plan_name + '.plan')
    lines = bytes_of(plan_path).decode().splitlines()
    membership = {int(t[0]): [int(e) for e in t[5:]] for t in (line.split() for line in lines[1:] if line.strip())}
    require(all(len(es) == int(lines[layer+1].split()[4]) for layer,es in membership.items()), 'H1 plan count syntax')
    require(sum(map(len, membership.values())) == 35, f'{plan_name}: 35 plan experts')
    require({m['layer']: m['experts'] for m in replay['plan_membership']} == membership,
            f'{d.name}: saved runtime membership equals plan')
    rr = {'run': d.name, 'plan': plan_name, 'cells': []}
    for i in range(8):
        c = read_json(d / f'cell{i}.json')
        require(c == replay['cells'][i], f'{d.name}/cell{i}: cell and REPLAY raw records agree')
        require(c['ids'] == capture_cells[i]['ids'], f'{d.name}/cell{i}: frozen ordered IDs')
        gpu_slots = [j for j,e in enumerate(c['ids']) if e in membership[c['layer']]]
        require(c['gpu_slots'] == gpu_slots and c['matched_H'] == len(gpu_slots), f'{d.name}/cell{i}: GPU slot membership')
        saved = {}
        for mode in 'OCSP':
            path = d / f'cell{i}-{mode}.f32'
            binary_record(path, 6 * 5120)
            saved[mode] = bytes_of(path)
            output_bytes[(d.name, i, mode)] = saved[mode]
            h1['mode_output_files'] += 1
        require(saved['O'] == saved['C'] == capture_bytes[i], f'{d.name}/cell{i}: O=C=capture exact bytes')
        require(saved['S'] == saved['P'], f'{d.name}/cell{i}: S=P exact bytes in partition')
        cold = [j for j in range(6) if j not in gpu_slots]
        for j in cold:
            sl = slice(j*5120*4, (j+1)*5120*4)
            require(saved['P'][sl] == saved['O'][sl], f'{d.name}/cell{i}/slot{j}: cold output exact bytes')
            h1['cold_slot_checks'] += 1
        if i in (3,6):
            binary_record(d / f'cell{i}-reference.f32', 6*5120)
            h1['reference_files'] += 1
        steady = [s for s in c['samples'] if not s['first_use']]
        first = [s for s in c['samples'] if s['first_use']]
        require(len(steady)==48 and len(first)==4, f'{d.name}/cell{i}: 48 steady + 4 first-use')
        require({s['block'] for s in steady} == set(range(12)), f'{d.name}/cell{i}: all 12 blocks')
        by_block = {}
        for s in c['samples']:
            require(s['input_ready_ns'] < s['cpu_begin_ns'] < s['cpu_end_ns'] <= s['output_ready_ns'],
                    f'{d.name}/cell{i}/{s["mode"]}: host endpoint/CPU order')
            if s['mode']=='S' and c['matched_H']>0:
                require(s['submit_end_ns'] <= s['join_begin_ns'] < s['join_end_ns'] <= s['cpu_begin_ns'],
                        f'{d.name}/cell{i}: S joins before CPU')
            if s['mode']=='P' and c['matched_H']>0:
                require(s['submit_end_ns'] <= s['cpu_begin_ns'] < s['cpu_end_ns'] <= s['join_begin_ns'] < s['join_end_ns'] <= s['output_ready_ns'],
                        f'{d.name}/cell{i}: P joins after CPU')
            if not s['first_use']:
                by_block.setdefault(s['block'], {})[s['mode']] = s
        values = {k: [] for k in ['C_minus_P','S_minus_P','cpu_C_minus_P','P_extra_non_cpu_vs_C','P_submit','P_join','S_join']}
        blocks = []
        for b, m in sorted(by_block.items()):
            require(set(m)==set('OCSP') and {s['order'] for s in m.values()}==set(range(4)), f'{d.name}/cell{i}/block{b}: full four-mode block')
            total = {mode: s['output_ready_ns']-s['input_ready_ns'] for mode,s in m.items()}
            cpu = {mode: s['cpu_end_ns']-s['cpu_begin_ns'] for mode,s in m.items()}
            row = {'block': b, 'mode_order': [mode for mode,s in sorted(m.items(), key=lambda kv:kv[1]['order'])],
                   'C_minus_P': total['C']-total['P'], 'S_minus_P': total['S']-total['P'],
                   'cpu_C_minus_P': cpu['C']-cpu['P'],
                   'P_extra_non_cpu_vs_C': (total['P']-cpu['P'])-(total['C']-cpu['C']),
                   'P_submit': m['P']['submit_end_ns']-m['P']['cpu_set_end_ns'],
                   'P_join': m['P']['join_end_ns']-m['P']['join_begin_ns'],
                   'S_join': m['S']['join_end_ns']-m['S']['join_begin_ns']}
            require(row['C_minus_P'] == row['cpu_C_minus_P']-row['P_extra_non_cpu_vs_C'],
                    f'{d.name}/cell{i}/block{b}: exact elapsed decomposition')
            for k in values:
                values[k].append(row[k])
            blocks.append(row)
        cell_rec = {'cell': i, 'matched_H': c['matched_H'], 'gpu_experts': c['gpu_global_ids'],
                    'gpu_slots': gpu_slots, 'cold_slots': cold, 'blocks_ns': blocks,
                    'summaries': {k:describe_ns(v) for k,v in values.items()}}
        rr['cells'].append(cell_rec)
        h1['steady_observations'] += len(steady)
        h1['first_use_observations'] += len(first)
    h1['runs'].append(rr)
require(h1['cold_slot_checks']==150, '150 H1 cold-slot byte comparisons')
require(h1['steady_observations']==1536 and h1['first_use_observations']==128, 'H1 observation totals')
require([p['run'] for p in processes if p['family']=='h1']==[r['run'] for r in h1['runs']],
        'H1 run naming agrees with recorded chronological guard order')
require([r['plan'] for r in h1['runs']]==['H1_COMPACT','H1_NONCOMPACT','H1_NONCOMPACT','H1_COMPACT'],
        'H1 recorded chronological plan order')
for cell in range(8):
    compact = [r for r in h1['runs'] if r['plan']=='H1_COMPACT']
    noncompact = [r for r in h1['runs'] if r['plan']=='H1_NONCOMPACT']
    for paired in (compact, noncompact):
        for mode in 'OCSP':
            require(output_bytes[(paired[0]['run'],cell,mode)]==output_bytes[(paired[1]['run'],cell,mode)],
                    f'cell{cell}/{mode}: repeated process same-plan output bytes')
    a,b = [output_bytes[(rs[0]['run'],cell,'P')] for rs in (compact,noncompact)]
    different = [j for j in range(6) if a[j*20480:(j+1)*20480] != b[j*20480:(j+1)*20480]]
    expected = sorted(set(compact[0]['cells'][cell]['gpu_slots']) ^ set(noncompact[0]['cells'][cell]['gpu_slots']))
    require(different==expected, f'cell{cell}: across-partition P changes only expected reassigned slots')
    h1['partition_output_differences'][str(cell)]={'differing_slots':different,'expected_reassigned_slots':expected}
for cell in (1,4):
    pp = {}
    for plan in ('H1_COMPACT','H1_NONCOMPACT'):
        rows = [r['cells'][cell] for r in h1['runs'] if r['plan']==plan]
        pp[plan] = {'C_minus_P_process_medians_ms':[r['summaries']['C_minus_P']['median_ms'] for r in rows],
                    'S_minus_P_process_medians_ms':[r['summaries']['S_minus_P']['median_ms'] for r in rows],
                    'positive_C_minus_P_blocks':[r['summaries']['C_minus_P']['positive'] for r in rows]}
    pp['ratio_of_mean_process_saving_medians'] = st.mean(pp['H1_NONCOMPACT']['C_minus_P_process_medians_ms']) / st.mean(pp['H1_COMPACT']['C_minus_P_process_medians_ms'])
    h1['primary'][str(cell)] = pp


# 08: all five eight-graph samples retained; paired process median B/A, as designed.
cpu08 = {'runs': [], 'paired_ratios': {}, 'saved_output_files': 0, 'saved_output_f32_count':0,
         'timed_samples': 0, 'timed_graph_calls': 0, 'quality_path_records': [],
         'direct_rounding_comparison_records':0, 'sample_time_drift':[]}
for d in sorted((src / 'cpu08/runs').iterdir()):
    r = read_json(d / 'RUN.json')
    launch = read_json(d / 'LAUNCH.json')
    timing = r['spec']['stage']=='timing'
    require(len(r['blocks'])==3, f'{d.name}: three CPU08 cells')
    require(launch['environment']['LD_PRELOAD']=='' if timing else bool(launch['environment']['LD_PRELOAD']),
            f'{d.name}: recorded timing has empty preload; quality has interposer')
    rr = {'run':d.name, 'arm':r['spec']['mode'], 'cell_order':[], 'blocks':[]}
    for pos,b in enumerate(r['blocks']):
        i = b['cell']; c=capture_cells[i]
        require(b['ids']==c['ids'], f'{d.name}/cell{i}: frozen ordered IDs')
        require(b['x_FNV64']==fnv64(bytes_of(cap/c['x_file'])), f'{d.name}/cell{i}: saved x checksum equals actual capture bytes')
        require(b['threads']==12, f'{d.name}/cell{i}: recorded graph threads 12')
        require(b['worker_receipt_after_timing']['CPU']==list(range(12)), f'{d.name}/cell{i}: 12 recorded physical worker IDs')
        for w, projection in zip(b['weights'], c['projections']):
            require(w['ne']==projection['ne'] and w['name']==projection['name'] and w['type']==projection['type'],
                    f'{d.name}/cell{i}: recorded actual projection shapes/types')
            expected_compact=[eid for eid,flag in zip(c['ids'],projection['compact_mask']) if flag]
            require(w['compact_experts']==expected_compact, f'{d.name}/cell{i}: subset compact membership')
        file = d / b['output_file']
        binary_record(file, 6*5120)
        require(bytes_of(file)==capture_bytes[i], f'{d.name}/cell{i}: saved output equals capture bytes')
        cpu08['saved_output_files']+=1;cpu08['saved_output_f32_count']+=6*5120
        samples=b['samples']
        require(len(samples)==(5 if timing else 1), f'{d.name}/cell{i}: expected saved sample count')
        vals=[]
        for n,s in enumerate(samples):
            require(s['sample']==n and s['calls']==(8 if timing else 1) and isinstance(s['begin_ns'],int) and isinstance(s['end_ns'],int), f'{d.name}/cell{i}: integer sample with expected call count')
            require(s['begin_ns'] < s['end_ns'], f'{d.name}/cell{i}: positive sample time')
            require(s['output_FNV64']==fnv64(capture_bytes[i]), f'{d.name}/cell{i}: sample endpoint checksum equals actual capture')
            vals.append((s['end_ns']-s['begin_ns'])/s['calls'])
        if timing:
            require(all(a['end_ns']<=z['begin_ns'] for a,z in zip(samples,samples[1:])), f'{d.name}/cell{i}: sequential samples')
            cpu08['timed_samples']+=len(samples);cpu08['timed_graph_calls']+=sum(s['calls'] for s in samples)
            rr['cell_order'].append(i)
            rr['blocks'].append({'cell':i,'position':pos,'raw_samples':samples,
                                 'ms_per_graph':[v/1e6 for v in vals],
                                 'median_ms_per_graph':st.median(vals)/1e6,
                                 'planner_work_size_recorded':b['planner_work_size']})
            cpu08['sample_time_drift'].append({'run':d.name,'cell':i,'position':pos,
                    'first_sample_ms_per_graph':vals[0]/1e6,
                    'median_last_three_samples_ms_per_graph':st.median(vals[-3:])/1e6,
                    'first_over_last_three_median':vals[0]/st.median(vals[-3:]),
                    'scope':'descriptive diagnostic only; no samples excluded and no revised selection statistic'})
        else:
            cpu08['quality_path_records'].append({'run':d.name,'cell':i,**b['path_check']})
            for direct in b.get('direct_min128_checks',[]):
                require(len(direct['modes'])==4 and all(m['equal'] for m in direct['modes']), f'{d.name}/cell{i}: four rounding comparisons recorded equal')
                require(all(math.isfinite(struct.unpack('<f',struct.pack('<I',m['output_bits']))[0]) for m in direct['modes']), f'{d.name}/cell{i}: recorded direct comparison output finite')
                cpu08['direct_rounding_comparison_records']+=len(direct['modes'])
    if timing:
        cpu08['runs'].append(rr)
cpu08['runs'].sort(key=lambda r:r['run'])
require([p['run'] for p in processes if p['family']=='cpu08' and p['run'].startswith('timing-')] ==
        [r['run'] for r in cpu08['runs']], 'CPU08 run naming agrees with recorded chronological guard order')
require(cpu08['timed_samples']==120 and cpu08['timed_graph_calls']==960, 'CPU08 120 samples / 960 timed graph calls')
require(cpu08['saved_output_files']==30 and cpu08['saved_output_f32_count']==921600, 'CPU08 30 saved F32 outputs / 921600 floats')
require(cpu08['direct_rounding_comparison_records']==24, 'CPU08 24 recorded direct four-rounding comparisons')
pair_indices=[(0,1),(3,2),(5,4),(6,7)]
for cell in (1,5,6):
    pairs=[]
    for pair,(ai,bi) in enumerate(pair_indices):
        ar,br=cpu08['runs'][ai],cpu08['runs'][bi]
        a,b=[next(q for q in r['blocks'] if q['cell']==cell) for r in (ar,br)]
        require('control' in ar['run'] and 'candidate' in br['run'], 'CPU08 correct fixed arm pairing')
        require(a['position']==b['position'], f'cell{cell}/pair{pair}: same position')
        am,bm=a['median_ms_per_graph'],b['median_ms_per_graph']
        pairs.append({'pair':pair+1,'A_run':ar['run'],'B_run':br['run'],'A_first':ai<bi,'position':a['position'],
                      'A_median_ms_per_graph':am,'B_median_ms_per_graph':bm,'B_over_A':bm/am,'A_minus_B_ms_per_graph':am-bm})
    cpu08['paired_ratios'][str(cell)]={'pairs':pairs,'median_B_over_A':st.median(p['B_over_A'] for p in pairs),
           'median_A_minus_B_ms_per_graph':st.median(p['A_minus_B_ms_per_graph'] for p in pairs),
           'positive_pairs':sum(p['A_minus_B_ms_per_graph']>0 for p in pairs),
           'position_groups':{str(pos):[p['B_over_A'] for p in pairs if p['position']==pos] for pos in sorted({p['position'] for p in pairs})}}


result={'schema':1,'status':'all performed checks passed','commit':'169a2f7eab8d52071661f0da6859d38df4206f8b',
        'method':'New independent Python calculation from saved raw integer timestamps, JSON, F32/I32 bytes. No supplied verify/analyze code is imported or executed.',
        'boundaries':{'performed':'Local saved-byte comparisons, finite scans, metadata consistency checks, arithmetic from all saved timestamps.',
          'record_only':'Model/native weight hashes, DSO hashes and providers, worker affinity, interposer counters, direct rounding kernel execution, process exits and machine restoration are saved runtime records; absent model/DSO contents were not reexecuted or certified.',
          'not_performed':'No real-machine API, build, inference, benchmark, deployment, new hardware timing, or model/ELF execution.',
          'units':'FBC 768 timed T1 in 6 processes; H1 1536 steady mode observations plus 128 separate first-use observations in 4 processes; CPU08 120 timed eight-graph samples / 960 graph calls in 8 processes. These counts do not create that many independent fresh processes.',
          'quality_sample_units':'CPU08 also has one single-graph sample per quality process/cell: 6 recorded quality samples. These are excluded from the 120 timed samples / 960 graph calls and from performance summaries.',
          'output_sampling':'FBC quality 65 rows/case/arm and timing final row. H1 saved final mode output, not every timed call output. CPU08 one saved output/cell/process and recorded sample endpoint FNV, not 960 individually saved outputs.'},
        'process_records':processes,'fbc':fbc,'h1':h1,'cpu08':cpu08,
        'check_count':len(checks),'checks':checks,'input_file_identities':inputs}
(out/'INDEPENDENT_MEASUREMENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'input_files':len(inputs),
      'FBC':fbc['cases'],'H1':h1['primary'],'CPU08':cpu08['paired_ratios']},ensure_ascii=False,indent=2))
