"""Audit first5 training routes and form one fixed-budget proposal offline.

No validation/holdout route is used. Hit counts are not saved-time estimates.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse,array,hashlib,json,math
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(b):
    p=b['prefix_tokens'];n=len(b['T1']);pre=b['prefill_trace'];t=b['T1_trace']
    assert n in (4,64) and b['logits_shape']==[5,129280] and b['logical_prefill_output_positions']==[p-1]
    assert len(b['continuation_inputs'])==n and len(b['next_token_targets_prefix_and_all_T1'])==n+1 and len(b['next_token_targets_for_saved_rows'])==5
    assert b['next_token_targets_for_saved_rows']==b['next_token_targets_prefix_and_all_T1'][:5]
    assert b['next_token_targets_prefix_and_all_T1'][:-1]==b['continuation_inputs']
    for trace in [pre,t]:
        s=trace['stats'];assert not s['enabled'] and s['flags']==s['dropped_events']==0
        assert s['calls']==len(trace['calls']) and s['attempts']==len(trace['attempts']) and s['records']==len(trace['records'])
        for c in trace['calls']:assert c['flags']==c['public_or_driver_status']==0 and c['completed'] and c['synchronized']
        for a in trace['attempts']:assert a['flags']==a['graph_status']==0 and a['graph_returned'] and a['caller_completed'] and a['context']==1
    assert pre['stats']['calls']==pre['stats']['attempts']==1 and pre['stats']['records']==pre['stats']['complete_decode_attempts']==0 and not pre['stats']['observing']
    a=pre['attempts'][0];assert a['position']==0 and a['n_tokens']==p and a['id']==1 and a['ubatch']==0 and a['records']==a['layers']==0 and not a['observing']
    assert a['label']==pre['calls'][0]['label']==dict(request=0,sequence=0,decode_call=0,phase=1,logical_input_token=0)
    assert b['actual_ubatches']==b['expected_ubatches']==[p]
    assert t['stats']['observing'] and t['stats']['calls']==t['stats']['attempts']==t['stats']['complete_decode_attempts']==n and t['stats']['records']==37*n
    for i,(a,c,step) in enumerate(zip(t['attempts'],t['calls'],b['T1'])):
        label=dict(request=0,sequence=0,decode_call=i,phase=2,logical_input_token=i)
        assert a['label']==c['label']==label and a['position']==p+i and a['id']==i+1 and a['ubatch']==0 and a['n_tokens']==1 and a['scope_t1'] and a['observing']
        assert a['records']==37 and a['first_record']==37*i and a['layers']==(1<<37)-1 and c['first_attempt']==i and c['attempts']==1
        rows=t['records'][37*i:37*(i+1)];assert sorted(r['layer'] for r in rows)==list(range(37))
        for j,r in enumerate(rows,37*i):
            for k,v in dict(ordinal=j,attempt_id=i+1,request=0,sequence=0,decode_call=i,ubatch=0,logical_input_token=i,position=p+i,generation=a['generation'],context=1,phase=2,projection=1,n_tokens=1,flags=0).items():assert r[k]==v,(k,r[k],v)
            assert len(r['ids'])==len(set(r['ids']))==6 and all(type(e)is int and 0<=e<384 for e in r['ids'])
        assert step['position']==p+i and step['next_target_token_id']==b['next_token_targets_prefix_and_all_T1'][i+1] and step['decode_status']==0 and step['full_logits_finite'] and step['synchronized']
        assert step['boundaries_ns']==sorted(step['boundaries_ns'])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    root=a.run;d=json.loads((root/'RUN.json').read_text());h=json.loads((root/'health.json').read_text());design=json.loads((R/'pilot-capture/PILOT_DESIGN.json').read_text());tokens=json.loads((R/'PILOT_TOKENS.json').read_text());frozen={c['case_id']:c for c in tokens['cases']}
    assert sha(R/'PILOT_TOKENS.json')==design['all_token_manifest_sha256'] and tokens['passed']
    assert d['passed'] and d['session_closed_before_owner_destruction'] and h['status']=='exited' and h['child_returncode']==0 and not (root/'STOP_REQUESTED').exists()
    for name,digest in json.loads((root/'RESULTS_SHA256.json').read_text()).items():assert sha(root/name)==digest,name
    pin=json.loads((root/'PROVIDER_EXPECTED.json').read_text());assert d['providers']==pin['providers'] and d['active_backend_providers']==pin['active_backend_providers']
    assert d['cparams']['n_ubatch']==d['cparams']['n_batch']==1024 and d['cparams']['n_ctx']==131072
    assert d['completed_cases']==['short64',*design['case_ids']]
    frequency=Counter();training=[];sources={};controls=[]
    for name in d['completed_cases']:
        path=root/(name+'.json');b=json.loads(path.read_text());check(b)
        vals=array.array('f');vals.frombytes((root/b['logits_file']).read_bytes());assert len(vals)==5*129280 and all(math.isfinite(v) for v in vals)
        if name=='short64':assert b['T1_byte_equal_previous_F_before'];controls.append(name);continue
        f=frozen[name];assert f['split']=='train' and b['prefix_tokens']==f['prefix_count'] and b['continuation_inputs']==f['used_T1_ids'] and b['next_token_targets_prefix_and_all_T1']==f['continuation_first65_ids']
        actual=next(c for c in d['cases'] if c['case_id']==name);assert actual['full_tokens']==f['prefix_ids']+f['full_reference_continuation_ids']
        assert len(b['T1'])==64;sources[name]=dict(result_sha256=sha(path),logits_sha256=sha(root/b['logits_file']),prefix_tokens=f['prefix_count'])
        for row in b['T1_trace']['records']:
            frequency.update((row['layer'],e) for e in row['ids'])
        training.append(b)
    inventory={x['layer']:x for x in d['native_expert_inventory']};budget=design['bank_selection_budget_bytes'];assert budget==int(458.4375*1048576)
    ranked=sorted(frequency,key=lambda pair:(-Fraction(frequency[pair],inventory[pair[0]]['native_triplet_bytes']),pair[0],pair[1]))
    chosen=[];used=0
    for layer,expert in ranked:
        assert layer<37 and inventory[layer]['host'];n=inventory[layer]['native_triplet_bytes']
        if used+n<=budget:chosen.append((layer,expert));used+=n
    chosen.sort();bank=set(chosen)
    per_case=[];aggregate=Counter();all_miss_runs=Counter()
    for b in training:
        rows=b['T1_trace']['records'];hist=Counter();per_layer=[];remaining=[222]*64
        for layer in range(37):
            selected=[r for r in rows if r['layer']==layer];assert len(selected)==64
            hs=[sum((layer,e) in bank for e in row['ids']) for row in selected];hist.update(hs)
            run=0;misses=[]
            for i,v in enumerate(hs):
                remaining[i]-=v
                if v==0:run+=1
                elif run:misses.append(run);run=0
            if run:misses.append(run)
            all_miss_runs.update(misses);per_layer.append(dict(layer=layer,H_each=hs,all_miss_run_lengths=misses))
        aggregate.update(hist);per_case.append(dict(case_id=b['name'],family_id=frozen[b['name']]['family_id'],lane=frozen[b['name']]['language_or_content'],H_histogram=[hist[i] for i in range(7)],mean_H=sum(i*hist[i] for i in range(7))/2368,remaining_CPU_expert_slots_per_token=remaining,per_layer=per_layer))
    total=5*64*37;hits=sum(i*aggregate[i] for i in range(7));assert sum(aggregate.values())==total and sum(frequency.values())==total*6
    plan=[f'XEON_HOT_FFN_PLAN_V1 40 384 5120 2304 {len(chosen)} {used}']
    for layer in range(40):
        ids=[e for l,e in chosen if l==layer];types=[p['type'].upper() for p in inventory[layer]['projections']]
        plan.append(' '.join(map(str,[layer,*types,len(ids),*ids])))
    outdir=a.output.parent;planfile=outdir/'TRAIN5_CANDIDATE.plan';assert not planfile.exists();planfile.write_text('\n'.join(plan)+'\n')
    result=dict(schema=1,passed=True,analysis_source_sha256=sha(Path(__file__)),run=root.name,controls_excluded_from_selection=controls,case_order=design['case_ids'],cases=5,families=len({frozen[x]['family_id'] for x in design['case_ids']}),training_calls=320,records=total,expert_selection_slots=total*6,sources=sources,
                candidate=dict(status='Offline proposal fitted to first5 training cases only; not allocated or admitted',budget_bytes=budget,native_payload_bytes=used,count=len(chosen),layer_count=len({x[0] for x in chosen}),selection_rule=design['bank_selection_rule'],plan_file=planfile.name,plan_sha256=sha(planfile),selected=[dict(layer=l,expert=e,training_inclusion_count=frequency[l,e],native_triplet_bytes=inventory[l]['native_triplet_bytes']) for l,e in chosen]),
                training_only_geometry=dict(H_histogram=[aggregate[i] for i in range(7)],mean_H=hits/total,selected_slot_fraction=hits/(total*6),H0_fraction=aggregate[0]/total,H6_fraction=aggregate[6]/total,all_miss_run_histogram=dict(sorted(all_miss_runs.items())),miss_runs_reset_at_each_case_and_layer=True),per_case=per_case,
                ranked_training_pairs=[dict(layer=l,expert=e,count=frequency[l,e],native_triplet_bytes=inventory[l]['native_triplet_bytes']) for l,e in ranked],
                saved_time=None,GPU_ready=None,actual_candidate_admission=None,validation_or_holdout_routes_used=False,quality_acceptance=None,
                limits=['Hit coverage is fitted to these5 training variants in3 families, not a generalization score.', 'Native budget is not a runtime allocation or guard pass for this different bank layout.', 'CPU residual slot count is not CPU time; no fitted saved-time or throughput claim.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,training_calls=320,records=total,candidate={k:v for k,v in result['candidate'].items() if k!='selected'},training_only_geometry=result['training_only_geometry']),indent=2))
if __name__=='__main__':main()
