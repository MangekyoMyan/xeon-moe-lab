from pathlib import Path
import hashlib,json,statistics,time
R=Path(__file__).resolve().parent;L=R.parent;logs=L/'iq3-plan-20261103/execution'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['cache-graph-final','cache-sanitized-final','cache-loader-cpu-01','cache-loader-gpu-01','cache-loader-stream-01','cache-loader-discovery-final']:
 folder=logs/name;st=json.loads((folder/'status.json').read_text());assert st['exitcode']==0 and st['error'] is None,name
 rows=[json.loads(line) for line in (folder/'stdout.log').read_text().splitlines() if line.startswith('{')];assert rows and all(row.get('pass',False) for row in rows),name
 checks.append({'name':name,'status':str(folder/'status.json'),'rows':rows,'stdout_sha256':sha(folder/'stdout.log')})
real=json.loads((R/'REAL_GRAPH.json').read_text());assert len(real)==6
summary={}
for direction in ['up','down']:
 rows=[r for t in real if t['direction']==direction for r in t['rows'] if r.get('kind')=='real_graph' and r['threads']==12]
 assert len(rows)==3
 summary[direction]={'original_ms_median_of_runs':statistics.median(r['original_ms'] for r in rows),
  'compact_ms_median_of_runs':statistics.median(r['compact_ms'] for r in rows),
  'speedup_median_of_runs':statistics.median(r['speedup'] for r in rows),'speedup_min':min(r['speedup'] for r in rows),'speedup_max':max(r['speedup'] for r in rows)}
for t in real:assert t['status']['exitcode']==0 and t['status']['error'] is None and t['rows'][-1]['pass']
for p,h in json.loads((R/'BUILD.json').read_text())['sources'].items():assert sha(R/p)==h,p
for lib in json.loads((R/'PROFILES.json').read_text())['libraries']:assert sha(Path(lib['file']))==lib['sha256']
linker=(R/'server-linker.log').read_text()
for name in ['libllama.so.0','libggml-cpu.so.0']:assert 'calling init: '+str(R/'runtime'/name) in linker
assert 'calling init: '+str(L/'gpu-stream-20261103/runtime/libggml-base.so.0') in linker
assert 'CUDA0: Tesla PG500-216' in (R/'server-devices.log').read_text()
validation={'passed':True,'unix':time.time(),'build_sha256':sha(R/'BUILD.json'),'checks':checks,'real_graph':summary,
 'timing_test_source':str(R/'test-cache-timing.cpp'),'timing_test_source_sha256':sha(R/'test-cache-timing.cpp'),
 'scope':'CPU graph + synthetic CPU/GPU model integration; no full DeepSeek startup or full-model speed claim',
 'sanitized_runtime_sha256':sha(R/'san/libggml-cpu.so.0.28.0'),'sanitized_test_sha256':sha(R/'san/test-cache'),
 'test_source_sha256':sha(R/'test-cache.cpp'),'loader_source_sha256':sha(R/'test-loader.cpp')}
(R/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
manifest=json.loads((R/'PROFILES.json').read_text());manifest['validation_sha256']=sha(R/'VALIDATION.json');(R/'PROFILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
