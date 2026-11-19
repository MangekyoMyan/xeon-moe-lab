#!/usr/bin/env python3
"""Parent's separate saved-data calculation; no engine/build/API execution.

Run from any directory. Source and output are relative to this script unless
explicitly supplied. This does not import the other agents' analysis code.
"""
import argparse
import hashlib
import json
import statistics
from fractions import Fraction
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
p.add_argument('--out', type=Path)
a = p.parse_args()
root = a.root.resolve()
src = root / 'source'
dest = a.out or root / 'PARENT_RECOMPUTATION.json'
seen = {}

def raw(path):
    data = path.read_bytes()
    seen[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return data

def doc(path):
    return json.loads(raw(path))

def med(values):
    z = sorted(values)
    n = len(z)
    return Fraction(z[n // 2]) if n % 2 else Fraction(z[n // 2 - 1] + z[n // 2], 2)

def ms(x):
    return float(x / 1_000_000)

identities = doc(root / 'SOURCE_IDENTITIES.json')
for rec in identities['files']:
    data = raw(src / rec['path'])
    assert len(data) == rec['bytes']
    assert hashlib.sha256(data).hexdigest() == rec['sha256']
    head = f'blob {len(data)}\0'.encode()
    assert hashlib.sha1(head + data).hexdigest() == rec['git_blob_sha']
assert len(identities['files']) == 578

fbc_runs = [doc(d / 'RUN.json') for d in sorted((src / 'fbc/runs').glob('timing-*'))]
assert ''.join(r['arm'] for r in fbc_runs) == 'FBCCBF'
fbc = {}
for case in [b['case_id'] for b in fbc_runs[0]['blocks']]:
    times = []
    for run in fbc_runs:
        b = next(b for b in run['blocks'] if b['case_id'] == case)
        assert len(b['start_ns']) == len(b['output_ready_ns']) == 64
        times.append(Fraction(sum(b['output_ready_ns']) - sum(b['start_ns']), 64))
    pairs = {'B_minus_F': [times[1]-times[0], times[4]-times[5]],
             'C_minus_B': [times[2]-times[1], times[3]-times[4]]}
    fbc[case] = {'run_mean_ms': [ms(t) for t in times],
                 'paired_differences_ms': {k: [ms(t) for t in v] for k, v in pairs.items()},
                 'mean_paired_differences_ms': {k: ms(sum(v)/2) for k, v in pairs.items()},
                 'arm_mean_ms': {arm: ms(sum(t for t, r in zip(times, fbc_runs) if r['arm']==arm)/2) for arm in 'FBC'},
                 'F_later_minus_earlier_ms': ms(times[-1]-times[0])}

h1 = {}
for cell in (1, 4):
    rows = []
    for d in sorted((src / 'h1/runs').iterdir()):
        c = doc(d / f'cell{cell}.json')
        groups = {}
        for s in c['samples']:
            if not s['first_use']:
                groups.setdefault(s['block'], {})[s['mode']] = s
        assert len(groups) == 12
        differences = {'C_minus_P': [], 'C_minus_S': [], 'S_minus_P': []}
        for g in groups.values():
            assert set(g) == set('OCSP')
            durations = {m: s['output_ready_ns'] - s['input_ready_ns'] for m, s in g.items()}
            for name in differences:
                x, y = name.split('_minus_')
                differences[name].append(durations[x]-durations[y])
        assert all(t > 0 for t in differences['C_minus_P'])
        rows.append({'run': d.name, 'gpu_ids': c['gpu_global_ids'],
                     'median_ms': {k: ms(med(v)) for k, v in differences.items()},
                     'raw_differences_ns': differences})
    ratio = (rows[1]['median_ms']['C_minus_P'] + rows[2]['median_ms']['C_minus_P']) / (rows[0]['median_ms']['C_minus_P'] + rows[3]['median_ms']['C_minus_P'])
    h1[str(cell)] = {'runs': rows, 'saving_ratio_noncompact_to_compact': ratio}

cpu = []
for d in sorted((src / 'cpu08/runs').glob('timing-*')):
    r = doc(d / 'RUN.json')
    vals = {}
    for b in r['blocks']:
        ss = [Fraction(s['end_ns']-s['begin_ns'], s['calls']) for s in b['samples']]
        assert len(ss) == 5 and all(s['calls']==8 for s in b['samples'])
        vals[b['cell']] = {'median': med(ss), 'samples': ss}
    cpu.append((d.name, vals))
cpu_result = {}
for cell in (1,5,6):
    ratios = [cpu[bi][1][cell]['median'] / cpu[ai][1][cell]['median'] for ai,bi in [(0,1),(3,2),(5,4),(6,7)]]
    cpu_result[str(cell)] = {'paired_B_over_A': [float(t) for t in ratios],
                             'median_B_over_A': float(med(ratios)),
                             'all_ms_per_graph': {name:[ms(t) for t in v[cell]['samples']] for name,v in cpu}}
mx_drift = [v[6]['samples'][0] / med(v[6]['samples'][-3:]) - 1 for _,v in cpu]

# Independently follow the pre-allocation dependencies, and confirm placement
# and order after allocation; graph records are not device timelines.
graphs = []
for case in fbc:
    pre = doc(src / 'fbc/runs/quality-F' / f'{case}-graph-pre.json')
    post = doc(src / 'fbc/runs/quality-F' / f'{case}-graph-post.json')
    before = {t['id']:t for t in pre['tensors']}
    after = {t['id']:t for t in post['tensors']}
    nodes = {after[t]['name']:after[t] for t in post['nodes']}
    order = {t:i for i,t in enumerate(post['nodes'])}
    checked = []
    for layer in range(37):
        get = lambda n: nodes[f'{n}-{layer}']
        norm = get('ffn_norm')
        route = [get(n) for n in ('ffn_moe_gate','ffn_moe_up','ffn_moe_swiglu_limited','ffn_moe_down')]
        shared = [get(n) for n in ('ffn_gate','ffn_up','ffn_swiglu_limited','ffn_shexp')]
        assert all(t['assigned_backend']=='CPU' for t in route)
        assert all(t['assigned_backend']=='CUDA0' for t in shared)
        assert shared[0]['src'][1] == shared[1]['src'][1] == norm['id']
        route_input = before[before[route[0]['id']]['src'][1]]
        assert route_input['op'] == 'RESHAPE' and route_input['src'][0] == norm['id']
        assert before[route[0]['id']]['src'][1:3] == before[route[1]['id']]['src'][1:3]
        assert shared[2]['src'][:2] == [shared[0]['id'],shared[1]['id']]
        assert shared[3]['src'][1] == shared[2]['id']
        assert get('ffn_out')['src'][:2] == [get('ffn_moe_out')['id'],shared[3]['id']]
        assert max(order[t['id']] for t in route) < order[get('ffn_moe_out')['id']] < min(order[t['id']] for t in shared)
        weights = [after[t['src'][0]] for t in (shared[0],shared[1],shared[3])]
        assert all(t['buffer_type']=='CUDA0' for t in weights)
        checked.append(layer)
    graphs.append({'case':case,'checked_layers':checked,'layer_count':len(checked),
                   'conclusion':'Independent branches from same norm; existing shared weights CUDA0; shared nodes scheduled after CPU branch. No hardware timing or speed prediction.'})

result = {'commit':identities['commit'],'method':'Separate parent implementation; reads exact pinned raw inputs, no imports of provider or agent analysis.',
          'source_byte_identity':{'files':len(identities['files']), 'bytes':sum(r['bytes'] for r in identities['files']), 'sha256_and_git_blob_sha1_all_match':True},
          'FBC':fbc,'H1':h1,'CPU08':cpu_result,
          'MX_first_vs_last_three_median_excess_percent':{'min':float(min(mx_drift)*100),'max':float(max(mx_drift)*100),'all':[float(t*100) for t in mx_drift]},
          'shared_branch_graph_checks':graphs,'intake_record':doc(src/'INTAKE.json'),
          'limitations':['Saved-record arithmetic and graph evidence only. No real-machine model, ELF, inference or benchmark execution.', 'No new GPU timeline. Dependency independence alone does not quantify overlap benefit.'],
          'input_sha256':seen}
dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','source_files':len(identities['files']),'FBC':fbc,
                  'H1_saving_ratios':{k:v['saving_ratio_noncompact_to_compact'] for k,v in h1.items()},
                  'CPU08_ratio_medians':{k:v['median_B_over_A'] for k,v in cpu_result.items()},
                  'shared_branch_layers_per_case':[g['layer_count'] for g in graphs]},ensure_ascii=False,indent=2))
