#!/usr/bin/env python3
from pathlib import Path
from collections import Counter,defaultdict
from bisect import bisect_right
import argparse
import json
import re
from analyze_trace import LINE

def ns(seconds,fraction):return int(seconds)*10**9+int(fraction.ljust(9,'0')[:9])
def analyze(folder):
    a=json.loads((folder/'ANALYSIS.json').read_text());lo=a['ROI_start_ns'];hi=a['ROI_end_ns'];tid=a['decode_tid']
    stack=[];intervals=[]
    for line in (folder/'trace.txt').read_text().splitlines():
        m=LINE.match(line)
        if not m:continue
        pid,t,seconds,fraction,group,event,rest=m.groups()
        if int(t)!=tid:continue
        now=ns(seconds,fraction)
        if event=='cpu':stack.append(now)
        elif event=='cpu_ret' and stack:intervals.append((stack.pop(),now))
    intervals.sort();starts=[x[0] for x in intervals]
    pattern=re.compile(r'^\s*(\d+)/(\d+)\s+(\d+)\.(\d+):\s+cpu-clock[^:]*:\s+([0-9a-f]+)\s+(.*?)\s+\(([^)]*)\)\s*$')
    count=Counter();cross=defaultdict(Counter);threads=Counter();unparsed=0
    for line in (folder/'symbols.txt').read_text().splitlines():
        if 'cpu-clock' not in line:continue
        m=pattern.match(line)
        if not m:unparsed+=1;continue
        pid,t,seconds,fraction,ip,symbol,dso=m.groups();now=ns(seconds,fraction)
        if not lo<=now<hi:continue
        index=bisect_right(starts,now)-1
        where='inside_CPU_graph' if index>=0 and now<intervals[index][1] else 'outside_CPU_graph'
        if 'libgomp.so' in dso:label='OpenMP_runtime'
        elif 'compact_dot' in symbol:label='IQ3_compact_dot'
        elif symbol.startswith('ggml_vec_dot_'):label=symbol
        else:label='other_CPU'
        count[label]+=1;cross[label][where]+=1;threads[t]+=1
    if unparsed:raise RuntimeError(f'Unparsed CPU samples: {unparsed}')
    total=sum(count.values())
    result={'CPU_sample_count':total,'sample_period_ns':10000000,
      'scope':'Scheduled CPU time samples across all threads, not fractions of request wall time. Inside/outside class refers to simultaneous decode-thread CPU graph interval.',
      'classes':{k:{'samples':v,'CPU_sample_percent':100*v/total,**dict(cross[k])} for k,v in count.most_common()},
      'thread_sample_counts':dict(threads),'estimated_scheduled_CPU_seconds':total*.01}
    omp=cross['OpenMP_runtime'];result['OpenMP_samples_outside_CPU_graph_percent']=100*omp['outside_CPU_graph']/sum(omp.values())
    (folder/'SAMPLE_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(analyze(a.folder),indent=2))
