#!/usr/bin/env python3
"""Partition decode-thread host elapsed time; never label it GPU device timing."""
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_right
import argparse
import json
import re
import statistics
import subprocess

LINE=re.compile(r'^\s*(\d+)/(\d+)\s+(\d+)\.(\d+):\s+(\w+):(\w+):\s*(.*)$')
ARG=re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
ORDER=['CPU_graph','tensor_copy_service','CUDA_sync_outside_copy','GPU_backend_host_other','copy_preparation_other','other']

def category(name):
    if name=='cpu':return 'CPU_graph'
    if name=='copy':return 'tensor_copy_service'
    if name.startswith('sync_'):return 'CUDA_sync_outside_copy'
    if name=='gpu':return 'GPU_backend_host_other'
    if name=='copy_input':return 'copy_preparation_other'
    return None

def analyze(folder):
    text_path=folder/'trace.txt'
    if not text_path.exists():
        p=subprocess.run(['perf','script','-i',str(folder/'perf.data'),'--ns','-F','pid,tid,time,event,trace','--show-lost-events'],capture_output=True,text=True,check=True)
        text_path.write_text(p.stdout);(folder/'perf-script.stderr').write_text(p.stderr)
    lines=text_path.read_text().splitlines();events=[];unparsed=[]
    for line in lines:
        m=LINE.match(line)
        if not m:
            if line.strip():unparsed.append(line)
            continue
        pid,tid,seconds,fraction,group,name,rest=m.groups()
        args={}
        for key,quoted,plain in ARG.findall(rest):
            if quoted:args[key]=quoted
            else:
                try:args[key]=int(plain,0)
                except ValueError:args[key]=plain
        events.append({'pid':int(pid),'tid':int(tid),'ns':int(seconds)*10**9+int(fraction.ljust(9,'0')[:9]),'name':name,'args':args})
    events.sort(key=lambda e:e['ns'])
    if any('LOST' in line.upper() for line in unparsed):raise RuntimeError('Lost perf records')
    starts=[e for e in events if e['name']=='decode' and e['args'].get('n_tokens')==1]
    boundary_source='process_ubatch entries with n_tokens=1'
    if not starts:
        # Early traces attached to the legacy llama_decode wrapper, which this
        # server bypasses. Layer-zero CPU expert entry recurs once per decode.
        manifest=json.loads((folder/'RESULT.json').read_text())
        if manifest.get('final',{}).get('tokens_predicted',0)<8:raise RuntimeError('No verified decode request')
        starts=[e for e in events if e['name']=='cpu' and e['args'].get('first')=='ffn_moe_gate-0']
        boundary_source='layer0 CPU FFN entry; legacy decode wrapper was not called; capture enabled after first generated token'
    tids={e['tid'] for e in starts}
    if len(tids)!=1 or len(starts)<3:raise RuntimeError('Insufficient/unexpected decode boundaries')
    tid=tids.pop();lo=starts[0]['ns'];hi=starts[-1]['ns'];steps=len(starts)-1
    selected=[e for e in events if e['tid']==tid]
    if any(e['name']=='decode' and e['args'].get('n_tokens')!=1 for e in selected if lo<=e['ns']<hi):
        raise RuntimeError('Prefill or another request entered decode ROI')
    stacks=defaultdict(list);spans=[];unmatched_returns=[]
    paired={e['name'][:-4] for e in selected if e['name'].endswith('_ret')}
    for e in selected:
        name=e['name']
        if name.endswith('_ret'):
            base=name[:-4]
            if stacks[base]:
                begin=stacks[base].pop();spans.append({'kind':base,'start':begin['ns'],'end':e['ns'],'args':begin['args']})
            else:unmatched_returns.append(e)
        elif name in paired:stacks[name].append(e)
    unmatched_inside=[e for e in unmatched_returns if lo<=e['ns']<hi]
    unmatched_entries_inside=[e for stack in stacks.values() for e in stack if lo<=e['ns']<hi]
    if unmatched_inside or unmatched_entries_inside:raise RuntimeError('Unmatched event inside complete decode window')
    clipped=[dict(s,start=max(lo,s['start']),end=min(hi,s['end'])) for s in spans if s['end']>lo and s['start']<hi]
    boundaries=defaultdict(Counter)
    for s in clipped:
        c=category(s['kind'])
        if c:
            boundaries[s['start']][c]+=1;boundaries[s['end']][c]-=1
    active=Counter();times=Counter();previous=lo
    for t in sorted(set(boundaries)|{lo,hi}):
        c=next((c for c in ORDER[:-1] if active[c]>0),'other');times[c]+=t-previous
        active.update(boundaries[t]);previous=t
    assert sum(times.values())==hi-lo
    timing={k:{'total_ms':times[k]/1e6,'ms_per_token_step':times[k]/1e6/steps,'wall_percent':100*times[k]/(hi-lo)} for k in ORDER}
    copies=[s for s in clipped if s['kind']=='copy'];copies.sort(key=lambda s:s['start']);copy_starts=[s['start'] for s in copies]
    transfers=defaultdict(lambda:{'calls':0,'bytes':0});names=defaultdict(lambda:{'calls':0,'wall_ms':0,'bytes_H2D':0,'bytes_D2H':0})
    for s in copies:
        key=s['args'].get('src','?')+' -> '+s['args'].get('dst','?')
        names[key]['calls']+=1;names[key]['wall_ms']+=(s['end']-s['start'])/1e6
    scoped=[e for e in selected if lo<=e['ns']<hi]
    for e in scoped:
        if e['name'] not in ['memcpy','memcpy2d']:continue
        a=e['args'];n=a.get('bytes',a.get('width',0)*a.get('height',0));kind=a.get('kind',-1)
        if not isinstance(n,int):raise RuntimeError('Memcpy byte fetch fault')
        direction={0:'H2H',1:'H2D',2:'D2H',3:'D2D',4:'Default'}.get(kind,'unknown')
        transfers[direction]['calls']+=1;transfers[direction]['bytes']+=n
        index=bisect_right(copy_starts,e['ns'])-1
        if index>=0 and e['ns']<copies[index]['end'] and direction in ['H2D','D2H']:
            c=copies[index];key=c['args'].get('src','?')+' -> '+c['args'].get('dst','?')
            names[key]['bytes_'+direction]+=n
    cpu_names=defaultdict(lambda:{'calls':0,'wall_ms':0})
    for s in clipped:
        if s['kind']=='cpu':
            name=s['args'].get('first','?');cpu_names[name]['calls']+=1;cpu_names[name]['wall_ms']+=(s['end']-s['start'])/1e6
    for v in transfers.values():v['bytes_per_token_step']=v['bytes']/steps;v['calls_per_token_step']=v['calls']/steps
    intervals=[(b['ns']-a['ns'])/1e6 for a,b in zip(starts,starts[1:])]
    result={'folder':str(folder),'decode_tid':tid,'boundary_source':boundary_source,'steps':steps,'ROI_start_ns':lo,'ROI_end_ns':hi,
       'ROI_wall_ms':(hi-lo)/1e6,'mean_step_ms':statistics.mean(intervals),'median_step_ms':statistics.median(intervals),
       'exclusive_host_time':timing,'runtime_memcpy_calls':dict(transfers),
       'event_counts':dict(Counter(e['name'] for e in scoped)),
       'copies_by_tensor':dict(sorted(names.items(),key=lambda kv:kv[1]['wall_ms'],reverse=True)),
       'CPU_graphs_by_first_tensor':dict(sorted(cpu_names.items(),key=lambda kv:kv[1]['wall_ms'],reverse=True)),
       'unparsed_lines':unparsed[:12],'unmatched_inside_ROI':0,
       'boundary_unmatched_returns':len(unmatched_returns),'boundary_unmatched_entries':sum(len(x) for x in stacks.values()),
       'scope':'Decode-thread host wall time, partitioned without nested double counting. CPU graph includes workers/barriers; sync includes waiting for prior GPU work. Not GPU device DMA/kernel time. Runtime byte counts may omit CUDA graph internal copies.'}
    (folder/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();r=analyze(a.folder)
    print(json.dumps({k:r[k] for k in ['steps','mean_step_ms','exclusive_host_time','runtime_memcpy_calls','unmatched_inside_ROI']},indent=2))
