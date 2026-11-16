#!/usr/bin/env python3
"""Recompute static CPU plan coverage and a hypothetical larger-layout frontier.

Only reads saved counts/plan. No inference, no runtime/profile mutation.
The hypothetical frontier maximizes training assignment coverage, not speed.
"""
import argparse,collections,hashlib,itertools,json
from pathlib import Path

TYPES={18:'iq3_xxs',21:'iq3_s',22:'iq2_s'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    e=a.snapshot/'snapshot/llm-lab/exploration-20261112'
    paths=[e/'training/20261112-075449/TRAINING_COUNTS.json',e/'plans/frequency19200.plan',e/'plans/FREQUENCY_MANIFEST.json']
    records=json.loads(paths[0].read_text())['records'];lookup={r['name']:r for r in records}
    lines=paths[1].read_text().splitlines();header=lines[0].split();assert header[0]=='XEON_IQ_CACHE_PLAN_V2'
    expected_tensors,expected_slots,expected_bytes=map(int,header[1:])
    planned={};current_by_type=collections.Counter();current_hits=0
    for line in lines[1:]:
        v=line.split();name,typ=v[:2];k,m,experts,n=map(int,v[2:6]);ids=list(map(int,v[6:]));r=lookup[name]
        assert len(ids)==n and len(set(ids))==n and k==r['k'] and m==r['rows'] and experts==r['experts']
        assert typ==TYPES[r['type']] and (k//256)*m==46080
        planned[name]=set(ids);current_by_type[typ]+=n;current_hits+=sum(r['counts'][i] for i in ids)
    assert len(planned)==expected_tensors and sum(current_by_type.values())==expected_slots
    items=[]
    for name in planned:
        r=lookup[name]
        for i,c in enumerate(r['counts']):items.append({'name':name,'expert':i,'count':c,'type':TYPES[r['type']]})
    rank=sorted(items,key=lambda x:(-x['count'],x['name'],x['expert']))
    denominator=sum(x['count'] for x in items)
    observed_calls={lookup[name]['calls'] for name in planned}
    assert len(observed_calls)==1
    observed_calls=observed_calls.pop()
    assert expected_bytes==expected_slots*46080*144
    baseline_hits=sum(x['count'] for x in rank[:expected_slots]);assert baseline_hits==current_hits
    curve=[]
    for slots in [8000,12000,14400,16000,17280,18000,19200,19418,20000,24000,30720]:
        used=rank[:slots];hits=sum(x['count'] for x in used)
        curve.append({'slots':len(used),'bytes':len(used)*46080*144,'training_hit_rate':hits/denominator,'training_assignments':hits})
    groups={10:[],11:[]}
    for item in items:groups[11 if item['type']=='iq2_s' else 10].append(item)
    for group in groups.values():group.sort(key=lambda x:(-x['count'],x['name'],x['expert']))
    prefix={unit:[0]+list(itertools.accumulate(x['count'] for x in group)) for unit,group in groups.items()}
    new_frontier=[]
    for budget in [expected_bytes,120*1024**3,128*1024**3]:
        units=budget//(46080*16)
        best=(-1,0,0)
        for n11 in range(min(len(groups[11]),units//11)+1):
            n10=min(len(groups[10]),(units-n11*11)//10)
            hits=prefix[11][n11]+prefix[10][n10]
            if hits>best[0]:best=(hits,n10,n11)
        hits,n10,n11=best
        chosen=groups[10][:n10]+groups[11][:n11]
        chosen_ids={(x['name'],x['expert']) for x in chosen}
        evicted=[x for x in items if x['expert'] in planned[x['name']] and (x['name'],x['expert']) not in chosen_ids]
        added=[x for x in chosen if x['expert'] not in planned[x['name']]]
        new_frontier.append({'budget_bytes':budget,'selected_160byte_projections':n10,'selected_176byte_projections':n11,
            'selected_projections':n10+n11,'payload_bytes':(n10*160+n11*176)*46080,
            'training_assignments':hits,'training_hit_rate':hits/denominator,
            'training_hit_rate_change_percentage_points':100*(hits-current_hits)/denominator,
            'evicted_current_projections':len(evicted),'evicted_training_assignments':sum(x['count'] for x in evicted),
            'newly_cached_projections':len(added),'newly_cached_training_assignments':sum(x['count'] for x in added),
            'net_added_uncached_assignments_per_observed_call':(current_hits-hits)/observed_calls})
    unchanged_selection_new_bytes=sum(n*(176 if typ=='iq2_s' else 160)*46080 for typ,n in current_by_type.items())
    slack=120*1024**3-expected_bytes
    available_kib=71738220 # capture/CURRENT_STATE.json hardware.memory_kib.MemAvailable
    state_path=a.snapshot/'review/capture/CURRENT_STATE.json'
    state=json.loads(state_path.read_text())
    assert state['hardware']['memory_kib']['MemAvailable']==str(available_kib)+' kB'
    paths.append(state_path)
    result={'scope':'Saved training census only; no heldout/live routing and no speed measurement or prediction.',
      'source_sha256':{str(x.relative_to(a.snapshot)):sha(x) for x in paths},
      'current':{'tensor_count':expected_tensors,'projection_count':expected_slots,'eligible_projection_count':len(items),
        'payload_bytes':expected_bytes,'configured_budget_bytes':120*1024**3,'selected_by_type':dict(current_by_type),
        'global_hard_cap_bytes':128*1024**3,'observed_calls_per_tensor':observed_calls,
        'training_hit_assignments':current_hits,'eligible_training_assignments':denominator,'training_hit_rate':current_hits/denominator},
      'current_layout_coverage_curve':curve,
      'hypothetical_unsigned_activation_layout':{'assumed_block_bytes':{'iq3_xxs':160,'iq3_s':160,'iq2_s':176},
        'unchanged_selection_payload_bytes':unchanged_selection_new_bytes,'unchanged_selection_payload_GiB':unchanged_selection_new_bytes/1024**3,
        'larger_than_128GiB_cap':unchanged_selection_new_bytes>128*1024**3,
        'exact_training_coverage_optimum_under_each_budget':new_frontier,
        'optimality_scope':'All K*M/256=46080; two costs 10 or11 units of737280 bytes; enumerate number of11-cost selections with sorted count prefix sums for each group. Maximizes saved training assignment count only.'},
      'mixed_layout_without_eviction_theoretical_payload_limit':{
        'configured_budget_slack_bytes':slack,'configured_budget_slack_GiB':slack/1024**3,
        'additional_bytes_per_IQ3_projection':46080*16,'additional_bytes_per_IQ2_projection':46080*32,
        'maximum_IQ3_only_upgrades':slack//(46080*16),'maximum_IQ2_only_upgrades':slack//(46080*32),
        'integer_constraint':'n_IQ3 + 2*n_IQ2 <= floor(slack/737280) = 1962',
        'limit':'Payload accounting only. A new API/plan format and mixed-layout dispatch are necessary; metadata and peak loader memory are not included. No deployment or live cache mutation was performed.'},
      'captured_memory_floor_context':{
        'MemAvailable_bytes':available_kib*1024,'MemAvailable_GiB':available_kib/1024**2,
        'loader_reserve_bytes':64*1024**3,'above_reserve_GiB':available_kib/1024**2-64,
        'unchanged_selection_extra_GiB':(unchanged_selection_new_bytes-expected_bytes)/1024**3,
        'limit':'One idle snapshot only. Reserve is checked before every allocation, not a guaranteed runtime minimum. Do not infer peak headroom or propose raising the cap.'},
      'limits':['Training selection and assessment use the same census, so these are in-sample coverage values.',
                'Census includes CPU single-token calls and possible prefill tails; does not cover resident GPU37-39.',
                'The larger unsigned-activation layout was a previous local prototype, not the current iq-rhs activation-sidecar experiment.',
                'Cold fallback cost, larger memory traffic, activation conversion, type-dependent speedups and contention are not included.',
                'No new configuration or deployment plan is generated.']}
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'output':str(a.output),'current':result['current'],'new_layout':result['hypothetical_unsigned_activation_layout']},ensure_ascii=False))
if __name__=='__main__':main()
