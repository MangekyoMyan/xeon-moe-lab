from pathlib import Path
import ast,hashlib,json,statistics,time
R=Path(__file__).resolve().parent;L=R.parent;P=L/'postcompact-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['multi-graph-01','multi-sanitized-01','multi-loader-policy-final','multi-loader-partial-final']:
 f=P/'execution'/name;s=json.loads((f/'status.json').read_text());assert s['exitcode']==0 and s['error'] is None,name
 rows=[json.loads(v) for v in (f/'stdout.log').read_text().splitlines() if v.startswith('{')];assert rows and all(row.get('pass') for row in rows),name
 checks.append({'name':name,'status_file':str(f/'status.json'),'stdout_sha256':sha(f/'stdout.log'),'rows':rows})
real=json.loads((R/'REAL_GRAPH.json').read_text());assert len(real['runs'])==14
assert real['CPU_library_sha256']==sha(R/'runtime/libggml-cpu.so.0.28.0')
assert real['source_sha256']==sha(R/'test-cache.cpp') and real['binary_sha256']==sha(R/'test-cache')
summary={}
for case in sorted({x['case'] for x in real['runs']}):
 tests=[t for t in real['runs'] if t['case']==case]
 for t in tests:assert t['status']['exitcode']==0 and t['status']['error'] is None and t['rows'][-1]['pass']
 rows=[x for t in tests for x in t['rows'] if x.get('kind')=='real_graph' and x['threads']==12]
 summary[case]={'runs':len(rows),'original_ms':statistics.median(x['original_ms'] for x in rows),'compact_ms':statistics.median(x['compact_ms'] for x in rows),'speedup_median':statistics.median(x['speedup'] for x in rows),'speedup_min':min(x['speedup'] for x in rows),'speedup_max':max(x['speedup'] for x in rows)}
b=json.loads((R/'BUILD.json').read_text())
for name,digest in b['sources'].items():assert sha(R/name)==digest,name
manifest=json.loads((R/'PROFILES.json').read_text())
for lib in manifest['libraries']:assert sha(Path(lib['file']))==lib['sha256']
linker=(R/'server-linker.log').read_text()
for name in ['libllama.so.0','libggml-cpu.so.0']:assert 'calling init: '+str(R/'runtime'/name) in linker
assert 'CUDA0: Tesla PG500-216' in (R/'server-devices.log').read_text()
selection=json.loads((R/'SELECTION.json').read_text());assert selection['96GiB']['selected_tensors']==40 and selection['96GiB']['additional_bytes']<96<<30
for p in R.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
validation={'passed':True,'unix':time.time(),'build_sha256':sha(R/'BUILD.json'),'checks':checks,'real_graph':summary,
 'sanitized_CPU_library_sha256':sha(R/'san/libggml-cpu.so.0.28.0'),'sanitized_graph_test_sha256':sha(R/'san/test-cache'),
 'scope':'CPU graph and mixed IQ-format synthetic model CPU/GPU execution; full DeepSeek multi96 load/speed pending user startup',
 'selection_policy_sha256':sha(R/'cache-policy.h'),'full_model_candidate_measured':False}
(R/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n');manifest['validation_sha256']=sha(R/'VALIDATION.json');(R/'PROFILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'passed':True,'real_graph':summary},indent=2))
