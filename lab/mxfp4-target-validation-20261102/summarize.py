from pathlib import Path
import csv,json,statistics
R=Path(__file__).resolve().parent
rows=[];suites=[];b_count=b_elements=0;pin_snapshots=0
for folder in sorted((R/'results').iterdir()):
 if not (folder/'manifest.json').exists():continue
 m=json.loads((folder/'manifest.json').read_text());assert m['passed'] and m['comparison_failures']==0
 strict='--strict-cpu' in m['common_args']
 for line in (folder/'summary.jsonl').read_text().splitlines():
  j=json.loads(line);assert j['passed']
  rows.append({'suite':folder.name,'strict_cpu':strict,'type':j['type'],'direction':'down' if j['down'] else 'up','k':j['k'],'m':j['m'],'tokens':j['tokens'],'experts':j['experts'],'topk':j['topk'],'threads':j['threads'],'routing':j['routing'],'expert_rows':json.dumps(j['expert_rows']), 'stock_ms':j['stock']['median_ms'],'candidate_ms':j['candidate']['median_ms'],'speed_ratio':j['speed_ratio'],'time_reduction_percent':100*(1-1/j['speed_ratio']),'observed_ratio_min':j['ratio_lower_extreme'],'observed_ratio_max':j['ratio_upper_extreme']})
 for line in (folder/'comparisons.jsonl').read_text().splitlines():
  j=json.loads(line);assert j['passed'] and j['finite'] and j['bit_exact']
  if j['variant']=='B':b_count+=1;b_elements+=j['elements']
 if strict:
  for run in m['runs']:
   lines=Path(run['stderr']).read_text().splitlines()
   snaps=[json.loads(line)['strict_worker_affinities'] for line in lines if line.startswith('{"strict_worker_affinities"')]
   assert len(snaps)==1
   masks=[x['cpus'] for x in snaps[0]]
   assert all(len(x)==1 for x in masks) and sorted(x[0] for x in masks)==m['requested_affinity'],(folder,masks)
   pin_snapshots+=1
 suites.append({'suite':folder.name,'cases':m['cases'],'processes':len(m['runs']),'strict_cpu':strict,'passed':True})
with (R/'TIMINGS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def get(suite,direction,tokens):return next(x for x in rows if x['suite']==suite and x['direction']==direction and x['tokens']==tokens)
scaling=[]
for direction in ['up','down']:
 for tokens in [8,17]:
  a=get('pinned-t6-balanced',direction,tokens);b=get('pinned-t12-physical',direction,tokens);c=get('pinned-t24-smt',direction,tokens)
  scaling.append({'direction':direction,'inputs_per_expert':tokens,'candidate_6physical_to12physical_speedup':a['candidate_ms']/b['candidate_ms'],'candidate_12physical_to24logical_speedup':b['candidate_ms']/c['candidate_ms']})
statuses=[json.loads(p.read_text()) for p in (R/'execution').glob('*/status.json')]
max_temp=max(s['cpu_max_C'] for x in statuses for s in [x['initial']]+x['samples'])
record={'date':'2026-11-02','status':'synthetic_target_tests_complete','suites':suites,'case_definitions_total':len(rows),'candidate_to_stock_comparisons':b_count,'candidate_to_stock_fp32_elements':b_elements,'candidate_mismatches':0,'strict_worker_snapshots_verified':pin_snapshots,'route_and_dispatch':json.loads((R/'diagnostics/RESULT.json').read_text()),'max_observed_cpu_C':max_temp,'thermal_guard_stops':sum(x['error'] is not None for x in statuses),'candidate_prefill_scaling':scaling,'limitations':['synthetic CPU MUL_MAT_ID; no complete DeepSeek model test','not model tokens/sec','clock not fixed','repeated weights; no full384-expert real-shape workload','one reproducible 12-thread6-expert3-row up regression remains','new CPU cannot be predicted by existing CPU SMT scaling'],'production_modified':False,'api_restart':'not performed; user-run long startups policy'}
(R/'SUMMARY.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k in ['case_definitions_total','candidate_to_stock_comparisons','candidate_to_stock_fp32_elements','candidate_mismatches','strict_worker_snapshots_verified','max_observed_cpu_C','thermal_guard_stops','candidate_prefill_scaling']},indent=2))
