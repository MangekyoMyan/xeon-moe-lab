"""Compare only matched synthetic requests; retain unpaired cases as such."""
from pathlib import Path
import json,statistics,hashlib
R=Path(__file__).resolve().parent
known={}
for arm,names in [('baseline',['baseline']),('cache120',['cache120','cache120-adopted'])]:
    known[arm]={}
    for name in names:
        p=R/'profiles'/(name+'.json')
        if p.exists():known[arm][hashlib.sha256(p.read_bytes()).hexdigest()]=json.loads(p.read_text())
groups={}
for path in (R/'runs').glob('*/RESULT.json'):
    j=json.loads(path.read_text())
    if j.get('error') or j['mode']!='plain':continue
    label=j['label']
    arm='baseline' if label=='baseline' or label.startswith('baseline-r') else 'cache120' if label.startswith('cache120-r') else None
    if arm is None:continue
    assert j['config_sha256'] in known[arm],('Unrecognized profile',path)
    conf=known[arm][j['config_sha256']]
    expected_runtime=R/'cache120/runtime' if arm=='cache120' else R.parent/'iq-cache-next-20261103/runtime'
    for name in ['libggml-cpu.so.0.28.0','libllama.so.0.8.0']:
        assert [x for x in j['libraries'] if Path(x).name==name]==[str(expected_runtime/name)],('Wrong loaded library',path,name)
    t=j['final']['timings']
    assert t['prompt_n']==j['context'] and t['cache_n']==0,('Prompt/cache count mismatch',path)
    assert t['predicted_n']==j['tokens'] and j['output_token_count']==j['tokens'],('Generation count mismatch',path)
    assert j['final'].get('stopped_limit') or j['final'].get('stop_type')=='limit',('Unexpected stop',path)
    assert j['request']['cache_prompt'] is False and j['request']['ignore_eos'] is True
    key=(j['workload'],j['context'],j['tokens'])
    groups.setdefault(key,{'baseline':[],'cache120':[]})[arm].append((path,j))
result={'comparisons':[],'unpaired':[],'scope':'Matched synthetic full-model API requests, cache_prompt=false; no claim of general workload coverage.'}
for key,arms in sorted(groups.items()):
    if not all(arms.values()):result['unpaired'].append({'workload':key,'counts':{k:len(v) for k,v in arms.items()}});continue
    records=[j for values in arms.values() for _,j in values]
    prompt_hashes={hashlib.sha256(json.dumps(j['request'],sort_keys=True).encode()).hexdigest() for j in records}
    assert len(prompt_hashes)==1,('Request mismatch',key)
    output_hashes={j['output_sha256'] for j in records};token_hashes={j['output_token_sha256'] for j in records}
    row={'workload':key[0],'context':key[1],'generated_tokens':key[2],
         'request_sha256':next(iter(prompt_hashes)),'output_identical':len(output_hashes)==1,
         'returned_tokens_identical':len(token_hashes)==1,'arms':{}}
    for arm,values in arms.items():
        decode=[j['final']['timings']['predicted_per_second'] for _,j in values]
        prefill=[j['final']['timings']['prompt_per_second'] for _,j in values]
        row['arms'][arm]={'count':len(values),'decode_tok_s':decode,'decode_median_tok_s':statistics.median(decode),
          'prefill_tok_s':prefill,'prefill_median_tok_s':statistics.median(prefill),'files':[str(p.relative_to(R)) for p,_ in values]}
    row['decode_speedup']=row['arms']['cache120']['decode_median_tok_s']/row['arms']['baseline']['decode_median_tok_s']
    row['prefill_speedup']=row['arms']['cache120']['prefill_median_tok_s']/row['arms']['baseline']['prefill_median_tok_s']
    result['comparisons'].append(row)
result['all_outputs_identical']=bool(result['comparisons']) and all(x['output_identical'] and x['returned_tokens_identical'] for x in result['comparisons'])
result['expected_workloads_present']={(x['workload'],x['context'],x['generated_tokens']) for x in result['comparisons']}=={('count',1024,96),('code',128,128),('japanese',128,96)}
result['required_repetitions_complete']=result['expected_workloads_present'] and all(x['arms']['baseline']['count']>=2 and x['arms']['cache120']['count']>=3 for x in result['comparisons'])
(R/'API_COMPARISON.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
for r in result['comparisons']:
    print(json.dumps({k:v for k,v in r.items() if k!='arms'},ensure_ascii=False))
print('Unpaired:',result['unpaired'])
