from pathlib import Path
import json
R=Path(__file__).resolve().parent;O=R/'cpu-coordination-mx/main-last'
def main():
    source=json.loads((O/'API_RUNS.json').read_text());assert source['complete'] and source['error'] is None
    arms={name:[] for name in ['main-first-control','main-last']}
    for row in source['runs']:
        data=json.loads(Path(row['result']).read_text());assert data['error'] is None
        assert data['output_token_count']==data['final']['timings']['predicted_n']==96
        arms[row['profile']].append((row,data))
    assert all(len(x)==4 for x in arms.values())
    assert len({tuple(d['libraries']) for entries in arms.values() for _,d in entries})==1
    rows=[]
    for workload in ['count','code']:
        a=[d for row,d in arms['main-first-control'] if row['workload']==workload];b=[d for row,d in arms['main-last'] if row['workload']==workload];assert len(a)==len(b)==2
        assert all(d['request']==a[0]['request'] for d in a+b)
        same=len({(d['output_sha256'],d['output_token_sha256']) for d in a+b})==1
        decode=lambda group:sum(d['final']['timings']['predicted_ms'] for d in group)
        pp=lambda group:sum(d['final']['timings']['prompt_ms'] for d in group)
        rows.append({'workload':workload,'control_tok_s':[d['final']['timings']['predicted_per_second'] for d in a],'candidate_tok_s':[d['final']['timings']['predicted_per_second'] for d in b],
                     'decode_gain_percent':(decode(a)/decode(b)-1)*100,'prefill_time_reduction_percent':(1-pp(b)/pp(a))*100,'same_output':same})
    total=lambda name,key:sum(d['final']['timings'][key] for _,d in arms[name])
    warm={}
    for profile in arms:
        labels=sorted({row['label'] for row,_ in arms[profile]});warm[profile]=[]
        for label in labels:
            matches=list((R/'runs').glob('*-'+label+'-warm-count-plain-c1024-n8/RESULT.json'));assert len(matches)==1
            d=json.loads(matches[0].read_text());warm[profile].append(d['final']['timings']['prompt_ms'])
    result={'complete':True,'all_outputs_identical':all(x['same_output'] for x in rows),'decode_gain_percent':(total('main-first-control','predicted_ms')/total('main-last','predicted_ms')-1)*100,
            'prefill128_time_reduction_percent':(1-total('main-last','prompt_ms')/total('main-first-control','prompt_ms'))*100,
            'prefill1024_time_reduction_percent':(1-sum(warm['main-last'])/sum(warm['main-first-control']))*100,'warmup_prompt_ms':warm,'rows':rows,
            'scope':'Same binary flag0/1,ABBA; two synthetic inputs x two; all worker CPU assignments rotate, so gain cannot be attributed solely to GPU NUMA locality'}
    (O/'LIVE_ABBA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['all_outputs_identical']:raise SystemExit('Placement comparison output mismatch')
if __name__=='__main__':main()
