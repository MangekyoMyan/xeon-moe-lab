#!/usr/bin/env python3
import hashlib,json,re,statistics
from pathlib import Path
from full_replica_experiment import LAB,ROOT,OUT,MODEL,BUILD
from summarize_bottleneck_observations import summarize

def series(prefix):
 rows=[]
 for p in sorted(OUT.glob(prefix+'-*/summary.json')):
  d=json.loads(p.read_text());b=d['benchmarks'][0]
  phases=[q['time'] for q in d['phases'] if re.search(r': generation run \d+/\d+',q['message'])]
  ss=[s['rss_kib']/2**20 for s in d['samples'] if phases and s['unix_time']>=phases[0]]
  rows.append({'name':p.parent.name,'mode':d['mode'],'tokens_per_second':b['avg_ts'],'stddev_within_process':b['stddev_ts'],'process_rss_during_generation_median_gib':statistics.median(ss) if ss else None,'peak_rss_gib':d['max_rss_gib'],'disk_read_bytes_total':d['max_disk_read_bytes'],'boot_id':d['boot_id'],'cpu_library_sha256':d['cpu_library_sha256'],'threads':b['n_threads'],'repetitions':len(b['samples_ts']),'tokens_per_repetition':b['n_gen'],'parent_thp':d.get('parent_thp_enabled')})
 def group(mode):
  r=[q for q in rows if q['mode']==mode]
  if not r:return None
  rates=[q['tokens_per_second'] for q in r]
  return {'processes':len(r),'median_ts':statistics.median(rates),'mean_ts':statistics.mean(rates),'min_ts':min(rates),'max_ts':max(rates),'median_generation_rss_gib':statistics.median(q['process_rss_during_generation_median_gib'] for q in r),'peak_rss_gib':max(q['peak_rss_gib'] for q in r)}
 a=group(2);b=group(1)
 return {'runs':rows,'single_interleaved':a,'full_local':b,'median_gain_percent':100*(b['median_ts']/a['median_ts']-1) if a and b else None,'same_boot':len({q['boot_id'] for q in rows})==1,'same_cpu_library':len({q['cpu_library_sha256'] for q in rows})==1,'all_process_disk_reads_zero':all(q['disk_read_bytes_total']==0 for q in rows)}

def responses(path):
 d=json.loads(path.read_text());rs=d['responses']
 result={'identical':d.get('responses_identical',d.get('responses_bit_identical',len({r['response'] for r in rs})==1)),'variants':{}}
 for label in ('ABC' if any(x['label']=='C' for x in rs) else 'AB'):
  r=[x for x in rs if x['label']==label]
  result['variants'][label]={'processes':len(r),'eval_tokens':list({x['eval_tokens'] for x in r}),'median_timings_ms':{phase:statistics.median(x['timings_ms'][phase] for x in r) for phase in ['load','prompt eval','eval']},'all_timings_ms':[x['timings_ms'] for x in r]}
 a,b=[result['variants'][x]['median_timings_ms'] for x in 'AB']
 if 'C' in result['variants']:
  c=result['variants']['C']['median_timings_ms']
  result['fine_interleave_generation_gain_percent']=100*(a['eval']/c['eval']-1)
  result['full_vs_fine_interleave_generation_gain_percent']=100*(c['eval']/b['eval']-1)
 result['generation_gain_percent']=100*(a['eval']/b['eval']-1)
 result['warm_request_latency_reduction_percent']=100*(1-(b['prompt eval']+b['eval'])/(a['prompt eval']+a['eval']))
 return result

report={'model':str(MODEL),'main':series('confirm'),'without_thp':series('no-thp'),'sweep':{},'locality':[]}
for label in 'ABC':
 d=json.loads((OUT/f'sweep-{label}/summary.json').read_text())
 report['sweep'][label]={'mode':d['mode'],'rates':{b['n_threads']:b['avg_ts'] for b in d['benchmarks']}}
for p in sorted(OUT.glob('locality-*/observations.json')):report['locality'].append(summarize(p.parent))
for name,path in [('short_response',OUT/'validation/comparison.json'),('long_response',OUT/'responses-long/comparison.json')]:
 if path.exists():report[name]=responses(path)
report['placement']=json.loads((OUT/'placement-verified/placement.json').read_text())
report['correctness']=json.loads((OUT/'validation/comparison.json').read_text())
(OUT/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k in ['main','without_thp','short_response','long_response']},ensure_ascii=False,indent=2))
