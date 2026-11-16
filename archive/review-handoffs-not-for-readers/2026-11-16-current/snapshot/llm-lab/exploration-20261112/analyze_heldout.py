"""Compare frozen heldout result files without combining unrelated speedup percentages."""
from pathlib import Path
import argparse,json,statistics
R=Path(__file__).resolve().parent
def load_label(label):
    matches=list((R/'heldout').glob('*-'+label+'/RESULT.json'))
    if len(matches)!=1:raise RuntimeError(f'Expected exactly one result for {label}, got {len(matches)}')
    data=json.loads(matches[0].read_text())
    assert data['complete'] and data['error'] is None
    data['_file']=str(matches[0]);return data
def main():
    p=argparse.ArgumentParser();p.add_argument('--control',nargs='+',required=True);p.add_argument('--candidate',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--scope',required=True);p.add_argument('--final',action='store_true');a=p.parse_args()
    arms=[[load_label(x) for x in names] for names in [a.control,a.candidate]]
    all_runs=arms[0]+arms[1]
    assert len({x['_file'] for x in all_runs})==len(all_runs)
    assert len(arms[0])==len(arms[1])
    if a.final:
        assert len(arms[0])==len(arms[1])==2
        assert [arm for _,arm in sorted((x['started_unix'],arm) for arm in [0,1] for x in arms[arm])]==[0,1,1,0]
        assert all(x['token_collection']=='Complete generated token IDs from native nonstream final response' for x in all_runs)
        assert all(c['output_token_count']==c['timings']['predicted_n']==128 for x in all_runs for c in x['cases'])
    assert len({x['input_sha256'] for x in all_runs})==1
    names=[x['name'] for x in all_runs[0]['cases']]
    assert all([x['name'] for x in run['cases']]==names for run in all_runs)
    assert all(len({x['config_sha256'] for x in arm})==1 for arm in arms)
    timing_labels={x.get('ttft_measurement','Observed streaming wall time to first received content/token') for x in all_runs}
    assert len(timing_labels)==1
    def speed(cases):
        return sum(c['timings']['predicted_n']-1 for c in cases)*1000/sum(c['timings']['predicted_ms'] for c in cases)
    rows=[]
    for i,name in enumerate(names):
        cases=[[run['cases'][i] for run in arm] for arm in arms];flattened=cases[0]+cases[1]
        assert len({c['timings']['predicted_n'] for c in flattened})==1
        assert len({c['prompt_tokens'] for c in flattened})==1
        rows.append({'name':name,'control_tok_s':[x['timings']['predicted_per_second'] for x in cases[0]],'candidate_tok_s':[x['timings']['predicted_per_second'] for x in cases[1]],'gain_percent':(speed(cases[1])/speed(cases[0])-1)*100,'same_output':len({(x['output_sha256'],x['token_sha256']) for x in flattened})==1})
    totals=[]
    for arm in arms:
        cases=[c for run in arm for c in run['cases']]
        totals.append({'decode_tok_s':speed(cases),'sum_predicted_ms':sum(c['timings']['predicted_ms'] for c in cases),'mean_ttft_s':statistics.mean(c['ttft_s'] for c in cases),'mean_elapsed_s':statistics.mean(c['elapsed_s'] for c in cases),'case_count':len(cases)})
    same=all(x['same_output'] for x in rows)
    result={'scope':a.scope,'ttft_measurement':next(iter(timing_labels)),'sources':{'control':[x['_file'] for x in arms[0]],'candidate':[x['_file'] for x in arms[1]]},'rows':rows,'aggregate':{'control':totals[0],'candidate':totals[1]},'aggregate_gain_percent':(totals[1]['decode_tok_s']/totals[0]['decode_tok_s']-1)*100,'all_outputs_identical':same,'comparison_accepted':same,'input_sha256':all_runs[0]['input_sha256']}
    path=R/a.out;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if a.final and not same:raise SystemExit('Final comparison output mismatch; saved diagnostic, comparison not accepted')
if __name__=='__main__':main()
