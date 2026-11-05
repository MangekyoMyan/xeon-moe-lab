from pathlib import Path
import hashlib,json,subprocess,time
R=Path(__file__).resolve().parent;L=R.parent;guard=L/'postcompact-20261103/guard.py'
cases=json.loads((L/'expert-validation-20261101/cases.json').read_text())
manifest={'binary_sha256':hashlib.sha256((R/'test-cache').read_bytes()).hexdigest(),
 'CPU_library_sha256':hashlib.sha256((R/'runtime/libggml-cpu.so.0.28.0').read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256((R/'test-cache.cpp').read_bytes()).hexdigest(),'runs':[]}
for rep in range(3):
 selected=[c for c in cases if c['type'] in ['IQ2_S','IQ3_S'] or (rep==0 and c['type']=='IQ3_XXS')]
 if rep%2:selected.reverse()
 for c in selected:
  name='multi-real-'+c['name'].lower()+'-'+str(rep)
  cmd=['python3',str(guard),'--name',name,'--timeout','90','--','numactl','--interleave=0,1',str(R/'test-cache'),c['shard'],c['tensor']]
  subprocess.run(cmd,check=True);f=guard.parent/'execution'/name
  rows=[json.loads(x) for x in (f/'stdout.log').read_text().splitlines() if x.startswith('{')]
  st=json.loads((f/'status.json').read_text());assert st['exitcode']==0 and st['error'] is None and rows[-1]['pass']
  manifest['runs'].append({'name':name,'case':c['name'],'type':c['type'],'direction':c['part'],'rows':rows,'status':st})
  (R/'REAL_GRAPH.json').write_text(json.dumps(manifest,indent=2)+'\n')
  for row in rows:
   if row.get('kind')=='real_graph' and row['threads']==12:print(json.dumps({'case':c['name'],**row}),flush=True)
print('Real-weight graph comparisons complete',flush=True)
