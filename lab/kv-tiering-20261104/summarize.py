from pathlib import Path
import hashlib,json,statistics
R=Path(__file__).resolve().parent;L=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return [json.loads(x) for x in (R/'execution'/name/'stdout.log').read_text().splitlines() if x.startswith('{')]
def status(name):
 s=json.loads((R/'execution'/name/'status.json').read_text());assert s['exitcode']==0 and s['error'] is None;return s
memory=rows('memory-layout');status('memory-layout')
real=next(x for x in memory if x['real_KV_allocation']);estimate=next(x for x in memory if x['context']==131072 and not x['real_KV_allocation'])
for key in ['raw_bytes','CSA_bytes','HCA_bytes','LID_bytes','CSA_state_bytes','HCA_state_bytes','LID_state_bytes','total_bytes']:assert real[key]==estimate[key]
capacity=[]
for x in memory:
 if x['real_KV_allocation']:continue
 c=x['CSA_bytes']+x['HCA_bytes'];capacity.append({'context':x['context'],'total_KV_state_MiB':x['total_bytes']/2**20,'compressed_KV_MiB':c/2**20,'index_kept_on_GPU_MiB':c/5/2**20,'payload_reclaim_upper_MiB':c*4/5/2**20})
tr=json.loads((R/'TRANSFER_RUNS.json').read_text());assert tr['source_sha256']==sha(R/'tiering_probe.cpp') and tr['binary_sha256']==sha(R/'tiering_probe')
metrics={}
for pattern in ['spread','clustered']:
 metrics[pattern]={}
 for method in range(4):
  xs=[]
  for run in tr['runs']:
   status(run['name'])
   def at(n,m):return next(x['median_pair_ms'] for x in run['rows'] if x['rows']==n and x['method']==m and x['pattern']==pattern)
   cost=3*at(65536,method)+5*at(131072,method);base=3*at(65536,0)+5*at(131072,0)
   xs.append({'weighted_eight_pairs_ms':cost,'extra_vs_resident_ms':cost-base})
  metrics[pattern][str(method)]={'runs':xs,'median_extra_ms':statistics.median(x['extra_vs_resident_ms'] for x in xs),
    'extra_ms_range':[min(x['extra_vs_resident_ms'] for x in xs),max(x['extra_vs_resident_ms'] for x in xs)]}
moe=rows('moe-budget-layout');status('moe-budget-layout')
assert moe[0]['budget_MiB']==320 and moe[0]['enabled_layers']==0
layers=json.loads((L/'expert-validation-20261101/expert-layer-bytes.json').read_text());cpu_layers=[v for k,v in layers.items() if int(k)<37]
summary={'status':'prototype_tests_complete_production_adoption_deferred','real_128k_memory':real,'capacity_table':capacity,
 'selected_payload_bytes_per_8_pairs':8*512*512*2,'selected_ids_bytes_per_8_pairs':8*512*4,
 'timing_scope':'3 CSA bank pair medians + 5 HCA bank pair medians; serial extrapolation, not measured full token time; 3 separate runs',
 'timings':metrics,'methods':{'0':'GPU resident gather + Attention','1':'CPU GGML get_rows + f16 cast + H2D + Attention','2':'CPU serial byte gather + H2D + Attention','3':'CPU 12-thread byte gather + H2D + Attention'},
 'expert_layer_GiB_range':[min(cpu_layers)/2**30,max(cpu_layers)/2**30],'moe_budget_allocations':moe,
 'source_reviewers':['Astra xhigh architecture','Terra measurement review'],'production_changed':False,'full_model_started':False,'full_model_tiering_speed_measured':False,
 'important_limits':['Synthetic selected indices, not measured model routing','Indexer and KV updates excluded','No prefill/state restore integration','No assumption about actual expert cache hit rate','320MiB alone cannot initialize this built-in expert cache; combining with other free VRAM is a distinct untested option'],
 'initial_fixture_failure':'First fixture used640 keys, violating CUDA FA D512 padding. Corrected to768 (128 visible raw +128 masked padding +512 selected), supports_op checked; only the fixture process aborted.'}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'real_KV_state_MiB':real['total_bytes']/2**20,'reclaim_MiB':320,'timing_extra_spread_ms':metrics['spread']['3']['median_extra_ms'],'timing_extra_clustered_ms':metrics['clustered']['3']['median_extra_ms'],'expert_layer_GiB_range':summary['expert_layer_GiB_range'],'recommendation':'defer at current128k'},indent=2))
