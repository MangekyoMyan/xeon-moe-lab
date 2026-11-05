from pathlib import Path
from collections import Counter
import csv,json,re
R=Path(__file__).resolve().parent
rows=[];pmu=[];traces=[]
for folder in sorted((R/'operators').iterdir()):
    run=json.loads((folder/'RESULT.json').read_text());text=(folder/'stdout.txt').read_text()
    row={'folder':folder.name,'exit_code':run.get('exit_code'),'error':run.get('error'),'compute_buffer':run.get('compute_buffer',False)}
    for prefix in ['VALIDATION','TIMING']:
        line=next((x for x in text.splitlines() if x.startswith(prefix+' ')),None)
        if line:row.update(json.loads(line[len(prefix)+1:]))
    if 'TIMING ' in text:rows.append(row)
    if 'PMU_PASSES' in text:
        metrics={m.group(1):float(m.group(2)) for m in re.finditer(r'^([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)\s+([0-9.]+)\s*$',text,re.M)}
        if 'gpu__time_duration.sum' in metrics:
            duration=metrics['gpu__time_duration.sum']/1e9
            metrics['derived_HBM_read_plus_write_GB_s']=(metrics['dram__bytes_read.sum']+metrics['dram__bytes_write.sum'])/duration/1e9
        pmu.append({'folder':folder.name,'compute_buffer':run.get('compute_buffer',False),'metrics':metrics,'range_graph_repeats':3})
    p=folder/'activity.tsv'
    if p.exists():
        counts=Counter();duration=Counter();kernel_count=0;all_ns=0;starts=[];ends=[];dropped=0
        for line in p.read_text().splitlines():
            x=line.split('\t')
            if x[0]=='K':
                a,b=int(x[1]),int(x[2]);name=x[4];counts[name]+=1;duration[name]+=(b-a)/1e6;starts.append(a);ends.append(b);kernel_count+=1;all_ns+=b-a
            elif x[0]=='DROPPED':dropped+=int(x[1])
        traces.append({'folder':folder.name,'kernel_count':kernel_count,'kernel_sum_ms':all_ns/1e6,'kernel_timeline_span_ms':(max(ends)-min(starts))/1e6,'dropped':dropped,
                       'kernels':[{'name':k,'calls':counts[k],'sum_ms':v} for k,v in duration.most_common()]})
assert all(r['exit_code']==0 and not r['error'] and r['pass'] for r in rows)
assert all(t['dropped']==0 for t in traces)
keys=['folder','type','K','M','tokens','experts','skew','compute_buffer','stock_ms','candidate_ms','speedup','nmse','stock_ref_nmse','candidate_ref_nmse','max_abs','nonfinite','pass']
with (R/'operator-results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
result={'timing_cases':len(rows),'all_numeric_checks_passed':True,'timing_results':rows,'PMU':pmu,'GPU_activity':traces,
        'live_summary':'LIVE_SUMMARY.json','live_deep_summary':'LIVE_DEEP_SUMMARY.json','production_changed':False,
        'limitations':['No full-model candidate A/B or language-quality validation.','Standalone tests use up to96experts to preserve live API VRAM.','Graphs disabled for standalone comparisons; actual full-model graph impact unmeasured.','Routing is synthetic, uniform or skewed.','Candidate can select Q8_1 activation path instead of baseline F16, increasing numerical error.','Full-model hardware HBM/kernel timeline not collected; host uprobe timing is not DMA/kernel timing.']}
(R/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'timing_cases':len(rows),'pmu_cases':len(pmu),'activity_cases':len(traces),'all_numeric_checks_passed':True,'activity_dropped':sum(t['dropped'] for t in traces)},indent=2))
for p in pmu:print(p['folder'],p['metrics'].get('derived_HBM_read_plus_write_GB_s'))
