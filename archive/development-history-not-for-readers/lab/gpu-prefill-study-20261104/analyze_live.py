from pathlib import Path
from collections import defaultdict, Counter
import re,json,statistics,sys,bisect
R=Path(__file__).resolve().parent
pattern=re.compile(r'^\s*(\d+)/(\d+)\s+(\d+)\.(\d+):\s+(\w+):(\w+):\s*(.*)$')
argpat=re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')
deep='--deep' in sys.argv
folder=next(p for p in (R/'live').iterdir() if ('-trace-deep-' if deep else '-trace-') in p.name)
events=[]
for line in (folder/'trace.txt').read_text().splitlines():
    m=pattern.match(line)
    if not m:
        if line.strip():raise RuntimeError('Unparsed/lost trace line: '+line[:150])
        continue
    pid,tid,sec,frac,group,name,rest=m.groups();args={}
    for k,q,v in argpat.findall(rest):
        if q:args[k]=q
        else:
            try:args[k]=int(v,0)
            except ValueError:args[k]=v
    events.append({'tid':int(tid),'t':int(sec)*10**9+int(frac.ljust(9,'0')[:9]),'name':name,'args':args})
events.sort(key=lambda e:e['t']);paired={e['name'][:-4] for e in events if e['name'].endswith('_ret')}
stacks=defaultdict(list);spans=[];unmatched=[]
for e in events:
    name=e['name']
    if name.endswith('_ret'):
        key=(e['tid'],name[:-4])
        if stacks[key]:
            b=stacks[key].pop();spans.append({'tid':b['tid'],'name':key[1],'a':b['t'],'b':e['t'],'args':b['args']})
        else:unmatched.append(e)
    elif name in paired:stacks[(e['tid'],name)].append(e)
ub=[s for s in spans if s['name']=='ubatch'];assert ub and sum(s['args']['n_tokens'] for s in ub)==2048,ub
lo=min(s['a'] for s in ub);hi=max(s['b'] for s in ub);tid=ub[0]['tid'];assert all(s['tid']==tid for s in ub)
assert not [e for e in unmatched if lo<=e['t']<=hi]
assert not [e for seq in stacks.values() for e in seq if lo<=e['t']<=hi]
names=['cpu','copy','sync_stream','sync_event','sync_device','memcpy','memcpy2d','launch','memset','gpu','other']
bound=defaultdict(Counter)
for s in spans:
    if s['tid']!=tid or s['name'] not in names or s['b']<=lo or s['a']>=hi:continue
    bound[max(lo,s['a'])][s['name']]+=1;bound[min(hi,s['b'])][s['name']]-=1
active=Counter();elapsed=Counter();last=lo
for t in sorted(set(bound)|{lo,hi}):
    key=next((n for n in names if active[n]>0),'other');elapsed[key]+=t-last;active.update(bound[t]);last=t
scoped=[e for e in events if e['tid']==tid and lo<=e['t']<=hi]
scoped_times=[e['t'] for e in scoped]
all_stream_sync=sorted([s for s in spans if s['tid']==tid and s['name']=='sync_stream'],key=lambda x:x['a'])
stream_sync_times=[s['a'] for s in all_stream_sync]
transfers=defaultdict(lambda:dict(calls=0,bytes=0));mmid=[]
for e in scoped:
    if e['name'] in ['memcpy','memcpy2d']:
        a=e['args'];n=a.get('bytes',a.get('width',0)*a.get('height',0));k={0:'H2H',1:'H2D',2:'D2H',3:'D2D',4:'default'}.get(a.get('kind'),'unknown')
        transfers[k]['calls']+=1;transfers[k]['bytes']+=n
for s in spans:
    if s['name']=='mmid' and s['tid']==tid and lo<=s['a']<=s['b']<=hi:
        inside=scoped[bisect.bisect_left(scoped_times,s['a']):bisect.bisect_right(scoped_times,s['b'])]
        syncs=[e for e in inside if e['name']=='sync_stream_ret']
        sends=[e for e in inside if e['name']=='memcpy' and e['args'].get('kind')==1]
        cpu_sort=None
        if syncs and sends:
            send=next((e for e in sends if e['t']>syncs[0]['t']),None)
            if send:cpu_sort=(send['t']-syncs[0]['t'])/1e6
        nested_sync=all_stream_sync[bisect.bisect_left(stream_sync_times,s['a']):bisect.bisect_right(stream_sync_times,s['b'])]
        assert all(x['b']<=s['b'] for x in nested_sync)
        nested_sync.sort(key=lambda x:x['a'])
        mmid.append({'wall_ms':(s['b']-s['a'])/1e6,'between_first_sync_and_H2D_ms':cpu_sort,
                     'first_sync_ms':(nested_sync[0]['b']-nested_sync[0]['a'])/1e6 if nested_sync else 0,
                     'all_sync_ms':sum(x['b']-x['a'] for x in nested_sync)/1e6})
telemetry=[]
for file in sorted((R/'live').glob('*/RESULT.json')):
    d=json.loads(file.read_text());rows=[]
    for line in (file.parent/'dmon.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        cols=line.split()
        if len(cols)!=24:raise RuntimeError(('dmon columns',len(cols),cols))
        row={k:float(cols[i]) for k,i in [('power_W',3),('sm_time_percent',6),('memory_time_percent',7),('memory_clock_MHz',12),('SM_clock_MHz',13),('power_violation_percent',14),('thermal_violation',15),('PCIe_RX_MB_s',22),('PCIe_TX_MB_s',23)] if cols[i]!='-'}
        if row.get('sm_time_percent',0)>0:rows.append(row)
    stats={k:{'min':min(x[k] for x in rows),'median':statistics.median(x[k] for x in rows),'max':max(x[k] for x in rows)} for k in rows[0]} if rows else {}
    telemetry.append({'folder':file.parent.name,'mode':d['mode'],'timings':d['final']['timings'],'output_sha256':d['output_sha256'],'dmon_nonidle_samples':len(rows),'dmon_stats':stats})
result={'ubatches':[s['args']['n_tokens'] for s in sorted(ub,key=lambda s:s['a'])],'wall_ms':(hi-lo)/1e6,
        'exclusive_host_time':{k:{'ms':v/1e6,'percent':100*v/(hi-lo)} for k,v in elapsed.items()},
        'runtime_memcpy':dict(transfers),'mmid_calls':len(mmid),'mmid_inclusive_ms':sum(x['wall_ms'] for x in mmid),
        'mmid_sort_gap_ms':sum(x['between_first_sync_and_H2D_ms'] or 0 for x in mmid),
        'mmid_first_sync_ms':sum(x['first_sync_ms'] for x in mmid),'mmid_all_sync_ms':sum(x['all_sync_ms'] for x in mmid),
        'mmid_sort_gaps_found':sum(x['between_first_sync_and_H2D_ms'] is not None for x in mmid),'telemetry':telemetry,
        'scope':'Host spans; synchronization includes GPU/DMA completion. Not GPU hardware kernel/DRAM counters. Routing gap includes host overhead between observed boundaries.'}
copy_service=defaultdict(lambda:{'calls':0,'ms':0,'bytes':0})
for s in spans:
    if s['name']!='memcpy' or s['tid']!=tid or not lo<=s['a']<=s['b']<=hi:continue
    k={0:'H2H',1:'H2D',2:'D2H',3:'D2D',4:'default'}.get(s['args'].get('kind'),'unknown')
    v=copy_service[k];v['calls']+=1;v['ms']+=(s['b']-s['a'])/1e6;v['bytes']+=s['args'].get('bytes',0)
result['cuda_memcpy_API_service_by_direction']=dict(copy_service)
(R/('LIVE_DEEP_SUMMARY.json' if deep else 'LIVE_SUMMARY.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='telemetry'},indent=2))
