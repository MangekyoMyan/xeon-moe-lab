"""Compare identical teacher-forced prefixes without inventing omitted probabilities."""
from pathlib import Path
import argparse,json,math,statistics
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn'
def find(label):
    paths=list((OUT/'quality').glob('*-'+label+'/RESULT.json'));assert len(paths)==1
    d=json.loads(paths[0].read_text());assert d['complete'] and d['error'] is None;return paths[0],d
def finite(v):return isinstance(v,(int,float)) and math.isfinite(v)
def mean(values):return statistics.mean(values) if values else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',default='hot-ffn/QUALITY_COMPARISON.json');a=p.parse_args()
    ap,ad=find(a.control);bp,bd=find(a.candidate)
    assert ad['sequence_sha256']==bd['sequence_sha256'] and ad['input_sha256']==bd['input_sha256'] and ad['n_probs']==bd['n_probs']==100 and ad['steps_per_case']==bd['steps_per_case']
    assert [c['name'] for c in ad['cases']]==[c['name'] for c in bd['cases']]
    all_rows=[];cases=[]
    for ac,bc in zip(ad['cases'],bd['cases']):
        rows=[];assert len(ac['points'])==len(bc['points'])
        for x,y in zip(ac['points'],bc['points']):
            assert x['j']==y['j'] and x['known_next_id']==y['known_next_id']
            assert x['timings']['prompt_n']==y['timings']['prompt_n']==1 and x['timings']['cache_n']==y['timings']['cache_n']
            row={'j':x['j'],'greedy_match':x['generated_id']==y['generated_id'],'control_matches_teacher':x['matches_frozen_baseline'],'candidate_matches_teacher':y['matches_frozen_baseline'],
                 'control_T1_ms':x['timings']['prompt_ms'],'candidate_T1_ms':y['timings']['prompt_ms'],'both_probability_records':bool(x['probability'] and y['probability'])}
            if row['both_probability_records']:
                pa,pb=x['probability'],y['probability']
                for probability in [pa,pb]:
                    top=probability['top_logprobs'];assert len(top)==100 and len({v['id'] for v in top})==100
                    assert finite(probability.get('logprob')) and probability['logprob']<=1e-5
                    assert all(finite(v.get('logprob')) and v['logprob']<=1e-5 for v in top)
                ta={v['id']:v['logprob'] for v in pa['top_logprobs']};tb={v['id']:v['logprob'] for v in pb['top_logprobs']}
                common=ta.keys()&tb.keys();deltas=[abs(math.exp(ta[k])-math.exp(tb[k])) for k in common]
                target=x['known_next_id'];la=pa.get('logprob') if pa['id']==target else ta.get(target);lb=pb.get('logprob') if pb['id']==target else tb.get(target)
                row.update(common_top_ids=len(common),common_max_probability_delta=max(deltas,default=None),common_probability_L1=sum(deltas) if deltas else None,control_top_mass=sum(math.exp(v) for v in ta.values()),candidate_top_mass=sum(math.exp(v) for v in tb.values()),
                    teacher_logprob_delta=(lb-la) if finite(la) and finite(lb) else None,
                    raw_top1_match=max(ta,key=ta.get)==max(tb,key=tb.get) if ta and tb else None)
            rows.append(row)
        cases.append({'name':ac['name'],'points':rows,'greedy_agreement_fraction':sum(x['greedy_match'] for x in rows)/len(rows),'probability_pairs':sum(x['both_probability_records'] for x in rows)})
        all_rows+=rows
    n=len(all_rows);paired=[x for x in all_rows if x['both_probability_records']]
    la=[x['teacher_logprob_delta'] for x in paired if x.get('teacher_logprob_delta') is not None]
    result={'scope':'Same fixedprefixes,one newtoken perrequest. Rawtop100 not renormalized. Missing UTF8 probabilityrecords excluded;missing teacher outside top100 not assignedzero/-inf. Not PPL,KL,TVD,or qualitybenchmark.',
        'sources':{'control':str(ap),'candidate':str(bp)},'points':n,'greedy_agreement_fraction':sum(x['greedy_match'] for x in all_rows)/n,
        'control_self_replay_fraction':sum(x['control_matches_teacher'] for x in all_rows)/n,'candidate_teacher_agreement_fraction':sum(x['candidate_matches_teacher'] for x in all_rows)/n,
        'probability_pairs':len(paired),'teacher_logprob_pairs':len(la),'mean_teacher_logprob_delta':mean(la),'max_common_probability_delta':max((x['common_max_probability_delta'] for x in paired if x['common_max_probability_delta'] is not None),default=None),
        'mean_common_probability_L1':mean([x['common_probability_L1'] for x in paired if x['common_probability_L1'] is not None]),
        'mean_control_T1_ms':mean([x['control_T1_ms'] for x in all_rows]),'mean_candidate_T1_ms':mean([x['candidate_T1_ms'] for x in all_rows]),
        'T1_time_reduction_percent':(1-sum(x['candidate_T1_ms'] for x in all_rows)/sum(x['control_T1_ms'] for x in all_rows))*100,
        'timing_note':'Instrumented n_probs100 perrequest T1 prompt_ms; not continuous free-generation tok/s.', 'cases':cases}
    path=R/a.out;path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    assert result['control_self_replay_fraction']==1,'Baseline replay differs from its frozen sequence; interpret only after investigating'
if __name__=='__main__':main()
