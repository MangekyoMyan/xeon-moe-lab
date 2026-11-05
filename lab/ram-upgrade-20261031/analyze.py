#!/usr/bin/env python3
import collections
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

LAB=Path(__file__).resolve().parent.parent
OUT=LAB/'results/ram-upgrade-20261031'
sys.path[:0]=[str(LAB),str(LAB/'server-study-20261023')]
from summarize_quantize_tradeoff import matrix_summary,stock_summary,counters
from analyze_study import core_metrics,imc_metrics,io_values

def load(p): return json.loads(p.read_text())
def stats(v): return {'median':st.median(v),'min':min(v),'max':max(v),'n':len(v)}

oldmem=load(LAB/'results/server-study-20261023/summary.json')
oldquant=load(LAB/'results/quantize-tradeoff-20261021/REPORT.json')
oldlocal=load(LAB/'results/bottleneck-observation-20261019/streaming-reference.json')
data={'memory':{},'matvec':{},'stock':{},'counters':{},'local_read':{},'local_matvec':{}}
groups=collections.defaultdict(list)
for f in sorted((OUT/'memory/measurements').glob('*/run.json')):
    r=load(f); assert r['valid']
    core=core_metrics(f.parent/'core.csv');imc=imc_metrics(f.parent/'imc.csv')
    assert not core['major_faults']
    assert all(v==0 for v in r['thermal_throttle_delta'].values())
    assert all(v==0 for d in r['ECC_delta'].values() for v in d.values())
    assert io_values(r['process_io_after'])['read_bytes']==io_values(r['process_io_before'])['read_bytes']
    groups[r['label'].rsplit('_r',1)[0]].append({'logical_GBs':r['result']['logical_GBs'],
        'dependent_load_ns':r['result']['dependent_load_ns'],'IMC_read_GBs':imc['read_GBs'],
        'active_MHz':core['average_active_MHz'],'IPC':core['IPC'],
        'L2_pending_stall_fraction':core['L2_pending_stall_cycle_fraction'],
        'max_CPU_C':r['temperature_max_C'],'RAPL_package_plus_DRAM_W':r['power']['package_plus_dram_W'],
        'RAPL_DRAM_W':sum(d['watts'] for d in r['power']['domains'].values() if d['name']=='dram'),
        'channel_rates':imc['entries']})
for name,rows in groups.items():
    new={k:stats([r[k] for r in rows]) for k in rows[0] if k!='channel_rates'}
    old=oldmem['cases'][name];key='dependent_load_ns' if name.startswith('latency') else 'logical_GBs'
    data['memory'][name]={'before':old,'after':new,'after_over_before':new[key]['median']/old[key]['median'],
                          'runs':rows}
for f in sorted((OUT/'quant').glob('matvec-k*')):
    new=matrix_summary(f);old=oldquant['matrix_comparisons'][f.name]
    data['matvec'][f.name]={k:{'before':old[k],'after':v,
        'speedup':old[k]['seconds_per_matvec']/v['seconds_per_matvec']} for k,v in new.items()}
for f in sorted((OUT/'quant').glob('quant-perf-*')):
    new=stock_summary(f);old=oldquant['stock_test_results'][f.name]
    pairs=[]
    for n in new:
        o=next(o for o in old if (o['type'],o['operation'],o['values'])==(n['type'],n['operation'],n['values']))
        pairs.append({'before':o,'after':n,'speedup':o['avg cycles/32 vals']/n['avg cycles/32 vals']})
    data['stock'][f.name]=pairs
for f in sorted((OUT/'counters').glob('counters-*')):
    new=counters(f);kind=f.name.removeprefix('counters-');old=oldquant['counter_results'][kind]
    new['active_MHz']=new['raw_core_events']['cycles:u']/(new['raw_core_events']['task-clock']*1000)
    old['active_MHz']=old['raw_core_events']['cycles:u']/(old['raw_core_events']['task-clock']*1000)
    data['counters'][kind]={'before':old,'after':new}
for placement in ['local','interleave']:
    rows=[load(p) for p in sorted((OUT/'local').glob(f'read-{placement}-*/stdout.txt'))]
    if rows:
        new=stats([st.median(r['gb_s']) for r in rows]);old=oldlocal['median_of_run_medians_GB_s'][placement]
        data['local_read'][placement]={'before':old,'after':new,'speedup':new['median']/old,'runs':rows}
for f in sorted((OUT/'local').glob('matvec-local-*')):
    new=matrix_summary(f);threads=f.name.rsplit('-t',1)[1]
    mixed=data['matvec'][f'matvec-k4096-stream-t{threads}']
    data['local_matvec'][f.name]={k:{'after':v,'speedup_vs_new_interleaved':mixed[k]['after']['seconds_per_matvec']/v['seconds_per_matvec']} for k,v in new.items()}
before=load(OUT/'before.json')
dimms=[]
for block in before['dimms']['stdout'].split('Memory Device')[1:]:
    fields={x.strip().split(':',1)[0]:x.strip().split(':',1)[1].strip() for x in block.splitlines() if ':' in x}
    dimms.append({k:fields.get(k) for k in ['Locator','Size','Configured Memory Speed','Manufacturer','Part Number','Rank']})
data['DIMMs']=dimms
data['sampled_CPU_max_C']=max([r['temperature_max_C'] for r in [load(f) for f in (OUT/'memory/measurements').glob('*/run.json')]]+
    [v for f in (OUT/'quant').glob('*/run.json') for row in load(f)['observations'] for v in row['cpu_temperatures_C'].values()]+
    [v for f in (OUT/'local').glob('*/run.json') for row in load(f)['observations'] for v in row['cpu_temperatures_C'].values()])
if (OUT/'after.json').exists():
    after=load(OUT/'after.json');assert before['counters']==after['counters']
    data['health']={'ECC_and_throttle_unchanged':True,'fan_mode_unchanged':before['fan_mode']==after['fan_mode'],
                    'fan_duty_after':after['fan_duty_raw'],'bmc_SEL_before':before['bmc_sel'],'bmc_SEL_after':after['bmc_sel']}
data['all_phases_complete']=all((OUT/f'{s}-complete.json').exists() for s in ('memory','quant','counters','local'))
(OUT/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
for k,v in data['memory'].items():
    key='dependent_load_ns' if k.startswith('latency') else 'logical_GBs'
    print(k,round(v['before'][key]['median'],3),'->',round(v['after'][key]['median'],3),f"ratio {v['after_over_before']:.3f}")
for k,v in data['matvec'].items():
    print(k, {q:round(v[q]['speedup'],3) for q in ['q8_0','q6_K','q4_K','q3_K']})
for k,v in data['counters'].items():
    print('COUNTERS',k,{p:{key:round(v[p][key],4) for key in ['dram_read_GB_s','ipc','stall_l2_pending_fraction','active_MHz']} for p in ['before','after']})
print('LOCAL',{k:{n:v[n] for n in ['before','after','speedup']} for k,v in data['local_read'].items()})
print('all phases complete',data['all_phases_complete'])
