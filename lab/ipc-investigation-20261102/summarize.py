from pathlib import Path
import json,csv,statistics
R=Path(__file__).resolve().parent
runs=[json.loads(p.read_text()) for p in sorted((R/'runs').glob('t*/RESULT.json'))]
assert len(runs)==20
for row in runs:
 folder=R/'runs'/row['label']
 status=json.loads((folder/'status.json').read_text());assert status['exitcode']==0 and status['error'] is None
 records=[json.loads(x) for x in (folder/'stderr.log').read_text().splitlines() if x.startswith('{')]
 masks=next(x['strict_worker_affinities'] for x in records if 'strict_worker_affinities' in x)
 wanted=list(range(row['threads']))
 assert sorted(x['cpus'][0] for x in masks)==wanted and all(len(x['cpus'])==1 for x in masks)
 assert all(e['running_percent']==100 for e in row['events'].values())
summaries=[]
for threads,tokens in [(1,8),(12,8),(24,8),(12,1),(12,3)]:
 group=[x for x in runs if x['threads']==threads and x['tokens_per_expert']==tokens]
 assert len(group)==4 and len(set(x['output_fnv1a64'] for x in group))==1
 result={'threads':threads,'inputs_per_expert':tokens,'output_fingerprint_match':True}
 for variant in ['stock','mxfp4']:
  sub=[x for x in group if x['variant']==variant];assert len(sub)==2
  total=lambda event:sum(x['events'][event]['value'] for x in sub)
  result[variant]={'ipc':total('instructions:u')/total('cycles:u'),'ipc_run_range':[min(x['ipc'] for x in sub),max(x['ipc'] for x in sub)],'instructions_per_graph':statistics.mean(x['instructions_per_graph'] for x in sub),'cycles_per_graph':statistics.mean(x['cycles_per_graph'] for x in sub),'mean_graph_ms':statistics.mean(x['graph_mean_ms'] for x in sub),'median_of_process_graph_medians_ms':statistics.median(x['graph_median_ms'] for x in sub),'generic_cache_miss_percent':100*total('cache-misses:u')/total('cache-references:u'),'cache_misses_per_graph':statistics.mean(x['events']['cache-misses:u']['value']/len(json.loads((R/'runs'/x['label']/'stdout.jsonl').read_text().splitlines()[0])['times_ms']) for x in sub),'average_scheduled_threads_range':[min(x['average_scheduled_threads'] for x in sub),max(x['average_scheduled_threads'] for x in sub)],'context_switch_counts':[x['events']['context-switches']['value'] for x in sub]}
 a=result['stock'];b=result['mxfp4'];result['instruction_reduction_percent']=100*(1-b['instructions_per_graph']/a['instructions_per_graph']);result['cycle_reduction_percent']=100*(1-b['cycles_per_graph']/a['cycles_per_graph']);result['mean_time_reduction_percent']=100*(1-b['mean_graph_ms']/a['mean_graph_ms']);result['mean_time_speedup']=a['mean_graph_ms']/b['mean_graph_ms'];result['median_time_speedup']=a['median_of_process_graph_medians_ms']/b['median_of_process_graph_medians_ms'];result['variable_24thread_condition']=threads==24
 summaries.append(result)
status_files=[json.loads(p.read_text()) for p in (R/'runs').glob('t*/status.json')]
report={'completed_runs':20,'all_hardware_event_running_percent':100,'strict_worker_masks_verified':True,'process_cpu_clock_agreement_range':[min(x['perf_to_process_cpu_time_ratio'] for x in runs),max(x['perf_to_process_cpu_time_ratio'] for x in runs)],'group_results':summaries,'max_cpu_C':max(s['cpu_max_C'] for r in status_files for s in [r['initial']]+r['samples']),'api_stopped_or_restarted':False,'full_model_benchmark_started':False,'failed_smoke_excluded':'smoke-stock-t1-n8; ACK trailing NUL framing error; smoke2 passed','scope':'MXFP4 synthetic MoE up projection K5120 M2304 experts6/top6, userspace ROI only; actual old prefill measured separately'}
(R/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
fields=['label','variant','threads','tokens_per_expert','ipc','graph_median_ms','graph_mean_ms','instructions_per_graph','cycles_per_graph','cache_reference_event_miss_percent','average_scheduled_threads','perf_to_process_cpu_time_ratio']
with (R/'COUNTERS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in runs)
print(json.dumps(report,indent=2))
