#!/usr/bin/env python3
"""Read pinned public data only. No supplied code, model, ELF, or command execution."""
import argparse, collections, hashlib, json, pathlib, statistics

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=pathlib.Path, required=True)
ap.add_argument('--out', type=pathlib.Path, required=True)
args = ap.parse_args()
repo = args.repo.resolve()
root = repo / 'archive/review-handoffs-not-for-readers'
sources = {}
def read(path):
    data = path.read_bytes()
    sources[str(path.relative_to(repo))] = hashlib.sha256(data).hexdigest()
    return data.decode()
def js(path):
    return json.loads(read(path))

trade = js(root / '2026-11-17-parallel-pilot/incoming/gpu_workspace_review_2026-11-17/plans/BANK_TRADEOFF_RECOMPUTED.json')
plan = read(root / '2026-11-17-parallel-pilot/integrated/capacity.plan').splitlines()
parsed_original = set()
for line in plan[1:]:
    a = line.split()
    assert len(a) == 5 + int(a[4])
    parsed_original.update((int(a[0]), int(e)) for e in a[5:])
assert parsed_original == {(s['layer'], s['expert']) for s in trade['original']['selected']}

compact_path = root / '2026-11-16-current/snapshot/llm-lab/exploration-20261112/plans/frequency19200.plan'
compact = {}
for line in read(compact_path).splitlines()[1:]:
    a = line.split()
    compact[a[0]] = {'type':a[1], 'ids':set(map(int,a[6:]))}
    assert len(compact[a[0]]['ids']) == int(a[5])
assert sum(len(v['ids']) for v in compact.values()) == 19200

capture = root / '2026-11-17-gpu-workspace/pilot-capture/runs/20261117T035402Z-U1024-d22b3e04'
training = {name:js(capture / (name+'.json'))['T1_trace']['records'] for name in trade['training_case_order']}
frequency = collections.Counter((r['layer'],e) for records in training.values() for r in records for e in r['ids'])
inventory = js(root / '2026-11-17-parallel-pilot/replay/runs/20261117T053052Z-U1024-1563d936/RUN.json')['native_expert_inventory']
inventory = {e['layer']:e for e in inventory}
result = {'schema':'validation3-hotpath-offline-review-v1', 'commit':'37f6390c9a5bc594658127651e204ab11420cd71',
          'interpretation':'Existing training5 routes only. Eight-layer selected set is the prior frozen diagnostic proposal, not a new fitted plan or measured speed.', 'plans':{}}
for key in ('original','conditional_alternative'):
    selected = collections.defaultdict(set)
    selected_rows = []
    for s in trade[key]['selected']:
        l,e = s['layer'],s['expert']
        selected[l].add(e)
        assert s['count']==frequency[l,e]
        row = {'layer':l,'expert':e,'training_count':frequency[l,e], 'projections':[]}
        for p in inventory[l]['projections']:
            iq = p['type'] in ('iq2_s','iq3_xxs','iq3_s')
            state = e in compact[p['name']]['ids'] if iq else None
            row['projections'].append({'name':p['name'], 'type':p['type'], 'compact_in_published_plan':state})
        selected_rows.append(row)
    counts, per_case = [0]*7, {}
    for name, records in training.items():
        assert len(records)==37*64
        hist, active = [0]*7,[0]*7
        for r in records:
            assert len(r['ids'])==6 and len(set(r['ids']))==6
            h = sum(e in selected[r['layer']] for e in r['ids'])
            hist[h]+=1
            if selected[r['layer']]: active[h]+=1
        hits = sum(h*n for h,n in enumerate(hist))
        per_case[name] = {'hits':hits,'all_H':hist,'active_H':active, 'active_layer_count':sum(bool(v) for v in selected.values()),
                          'active_H0_fraction':active[0]/sum(active)}
        assert hist == trade[key]['per_case'][name]['all_H']
        assert active == trade[key]['per_case'][name]['active_H']
        for h,n in enumerate(hist): counts[h]+=n
    nbytes=sum(inventory[l]['native_triplet_bytes']*len(es) for l,es in selected.items())
    assert nbytes==trade[key]['native_payload_bytes']
    iq_rows = [p for s in selected_rows for p in s['projections'] if p['compact_in_published_plan'] is not None]
    result['plans'][key] = {'selected':selected_rows,'native_bytes':nbytes,'native_MiB':nbytes/1048576,
        'expert_count':sum(len(es) for es in selected.values()),'active_layer_count':sum(bool(es) for es in selected.values()),
        'iq_projection_slots':len(iq_rows),'iq_compact_slots':sum(p['compact_in_published_plan'] for p in iq_rows),
        'all_H':counts,'total_hits':sum(h*n for h,n in enumerate(counts)), 'per_case':per_case}

orig, alt = result['plans']['original'],result['plans']['conditional_alternative']
result['comparison'] = {'byte_delta':alt['native_bytes']-orig['native_bytes'], 'MiB_delta':alt['native_MiB']-orig['native_MiB'],
    'expert_delta':alt['expert_count']-orig['expert_count'], 'active_layer_delta':alt['active_layer_count']-orig['active_layer_count'],
    'total_hit_count_ratio':alt['total_hits']/orig['total_hits'], 'per_case':{}}
for name in training:
    a,b=orig['per_case'][name],alt['per_case'][name]
    result['comparison']['per_case'][name] = {'original_hits':a['hits'],'alternative_hits':b['hits'],'ratio':b['hits']/a['hits'],
        'original_active_H0':a['active_H'][0], 'alternative_active_H0':b['active_H'][0],
        'original_active_H0_rate':a['active_H0_fraction'], 'alternative_active_H0_rate':b['active_H0_fraction']}

replay = js(root / '2026-11-17-parallel-pilot/replay/runs/20261117T053052Z-U1024-1563d936/REPLAY.json')
cells = []
for c in replay['cells']:
    measurements={}
    for mode in 'OCSP':
        ss=[s for s in c['samples'] if s['mode']==mode and not s['first_use']]
        assert len(ss)==12
        def mean_delta(end,start):
            return statistics.mean((s[end]-s[start])/1000 for s in ss)
        measurements[mode]={'mean_total_us':mean_delta('output_ready_ns','input_ready_ns'),
            'mean_cpu_span_us':mean_delta('cpu_end_ns','cpu_begin_ns'),
            'mean_join_span_us':mean_delta('join_end_ns','join_begin_ns')}
        if mode!='O':
            measurements[mode].update({'mean_partition_us':mean_delta('partition_end_ns','input_ready_ns'),
                'mean_cpu_set_us':mean_delta('cpu_set_end_ns','partition_end_ns'),
                'mean_submit_span_us':mean_delta('submit_end_ns','cpu_set_end_ns')})
    cells.append({'cell':c['cell'],'layer':c['layer'],'ids':c['ids'],'H':c['matched_H'],
        'gpu_slots':c['gpu_slots'],'runtime_compact':[{'type':p['type'],'mask':p['compact_mask']} for p in c['compact']],
        'measurements':measurements, 'O_minus_P_us':measurements['O']['mean_total_us']-measurements['P']['mean_total_us'],
        'S_minus_P_us':measurements['S']['mean_total_us']-measurements['P']['mean_total_us']})
result['operator_cells']=cells
result['operator_warning']='O independent graph vs C private dispatcher, not whole-model F/B/C. CPU span C/S/P includes output copy; O does not. Join span is wait after CPU work, not GPU service time. Host timestamps not device profiler.'
result['paired_H1_proposed_cells']=[]
for ci,compact_e,native_e in [(1,331,85),(4,78,100)]:
    c=cells[ci];l=c['layer']
    assert compact_e in c['ids'] and native_e in c['ids']
    result['paired_H1_proposed_cells'].append({'source_cell':ci,'layer':l,'ids':c['ids'],'compact_expert':compact_e,'noncompact_expert':native_e,
        'compact_train_count':frequency[l,compact_e], 'noncompact_train_count':frequency[l,native_e],
        'naive_value_break_even_gain_ratio':frequency[l,compact_e]/frequency[l,native_e],
        'break_even_warning':'Only a frequency*per-hit-saving illustration. Joint selection, active-layer overhead, GPU completion, residual CPU mix and workload weighting must be included before selecting a full bank.'})
result['frequent_noncompact_capture_options']=[]
orig_map=collections.defaultdict(set)
for l,e in parsed_original: orig_map[l].add(e)
for l,e in [(3,147),(7,270)]:
    per_case={}
    for name, records in training.items():
        target=[r for r in records if r['layer']==l and e in r['ids']]
        joint=[r for r in target if any(v in orig_map[l] for v in r['ids'])]
        per_case[name]={'count':len(target),'joint_original_hot_count':len(joint),
            'first_joint_record':({k:joint[0][k] for k in ['ordinal','decode_call','position','ids']} if joint else None)}
    result['frequent_noncompact_capture_options'].append({'layer':l,'expert':e,'count':frequency[l,e],'per_case':per_case,
        'activation_status':'Not present in the eight saved activation cells. Requires dedicated baseline capture and observation on/off checks.'})
result['explicit_host_copy_bytes']={'per_layer_H0_input':5120*4,'per_layer_H0_ids':6*4,'per_layer_H0_output':5120*6*4,
    '23_layer_H0_total':23*(5120*4+6*4+5120*6*4), 'warning':'Explicit dispatcher copies only; not a measurement and excludes main scheduler transfers.'}
result['source_sha256']=sources
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(args.out),'comparison':result['comparison'], 'compact_summary':{k:{n:v[n] for n in ['expert_count','iq_projection_slots','iq_compact_slots','native_MiB']} for k,v in result['plans'].items()}, 'paired_H1_proposed_cells':result['paired_H1_proposed_cells']},ensure_ascii=False,indent=2))
