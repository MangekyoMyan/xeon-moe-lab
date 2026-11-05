#!/usr/bin/env python3
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
def read(label):return json.loads((R/'results'/label/'result.json').read_text())
def metric(label):
 j=read(label);s=j['stats'];return {'label':label,'prefill_s':s['XV_PREFILL'][0]['seconds'],
  'prefill_tps':s['XV_PREFILL'][0]['tps'],'steady_decode_tps':s['XV_TIMING'][0]['decode_tps'],
  'decode_seconds':s['XV_ALL_TIMING'][0]['decode_seconds'],'decode_steps':s['XV_ALL_TIMING'][0]['decode_steps'],'total_s':s['XV_ALL_TIMING'][0]['total_seconds'],
  'first_decode_s':s['XV_RECORD'][1]['seconds'],'io':s['XV_PREFILL_IO'][0]}
a=[metric(x) for x in ['final-a0','final-a1']];b=[metric(x) for x in ['final-b0','final-b1']]
ma=statistics.mean(x['prefill_s'] for x in a);mb=statistics.mean(x['prefill_s'] for x in b)
result={'scope':'GLM-4.5-Air Q8_0, 2048 fixed prompt tokens, 32 full-logit records; 131072 Q8 KV; 150W/ECC; 9GiB cache both arms',
 'order':['B0','A0','A1','B1'],'baseline':a,'candidate':b,
 'prefill_mean_s':{'baseline':ma,'candidate':mb},'time_reduction_percent':100*(1-mb/ma),'throughput_ratio':ma/mb,
 'steady_decode_mean_tps':{'baseline':statistics.mean(x['steady_decode_tps'] for x in a),'candidate':statistics.mean(x['steady_decode_tps'] for x in b)},
 'quality':{},'additional':{},'hardware':{}}
for n in ['final-b0','final-b1','quality-ja-b','short-b']:
 result['quality'][n]=json.loads((R/'results'/n/'logit-comparison.json').read_text())
for n in ['quality-ja-a','quality-ja-b','short-a','short-b']:result['additional'][n]=metric(n)
q=read('final-b0')['stats'];s=next(x for x in q['XV_STREAM'] if x['phase']=='prefill' and x['operations'])
result['candidate_prefill_breakdown']=s
result['candidate_worker_quantize_ms_including_small_decode_pool']=sum(x['quantize_ms'] for x in q['XV_EXACT'])
gpu=[];hbm=[];cpu=[];fan=[];vram=[]
for label in ['final-b0','final-a0','final-a1','final-b1','quality-ja-a','quality-ja-b','short-a','short-b']:
 j=json.loads((R.parent/'results'/R.name/label/'run.json').read_text())
 assert not j.get('error'),label
 for row in j['samples']:
  gpu.append(row['gpu']['gpu_C']);hbm.append(row['gpu']['hbm_C']);cpu.extend(row['cpu_C'].values());vram.append(row['gpu']['vram_used_MiB'])
  if 'fan_duty' in row:fan.extend(row['fan_duty']['percent'])
 result['hardware'][label]={'returncode':j['returncode'],'gpu_after':j['gpu_after'],'fan_after':j['fan_duty_after']}
result['v3']={}
for n in ['v3-short','v3-ja512','v3-long','ja1024-a','ja1024-b']:
 result['v3'][n]=metric(n)
 p=R/'results'/n/'logit-comparison.json'
 if p.exists():result['v3'][n]['quality']=json.loads(p.read_text())
 p=R/'results'/n/'v2-equivalence.json'
 if p.exists():result['v3'][n]['v2_equivalence']=json.loads(p.read_text())
 j=json.loads((R.parent/'results'/R.name/n/'run.json').read_text());assert not j.get('error'),n
 for row in j['samples']:
  gpu.append(row['gpu']['gpu_C']);hbm.append(row['gpu']['hbm_C']);cpu.extend(row['cpu_C'].values());vram.append(row['gpu']['vram_used_MiB'])
  if 'fan_duty' in row:fan.extend(row['fan_duty']['percent'])
 result['hardware'][n]={'returncode':j['returncode'],'gpu_after':j['gpu_after'],'fan_after':j['fan_duty_after']}
result['maxima']={'gpu_C':max(gpu),'hbm_C':max(hbm),'cpu_C':max(cpu),'fan_percent':max(fan),'vram_MiB':max(vram)}
if (R/'API_SMOKE.json').exists():
 api=json.loads((R/'API_SMOKE.json').read_text());result['api_smoke']={'success':api.get('success'),'error':api.get('error'),'supervisor_returncode':api.get('supervisor_returncode'),'requests':[{k:x.get(k) for k in ['name','first_content_s','total_s','content','timings','majflt','disk_read_bytes']} for x in api.get('requests',[])]}
(R/'FINAL_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['hardware','quality','additional','candidate_prefill_breakdown','api_smoke']},ensure_ascii=False,indent=2))
