"""Exact capacity/route-count plans from saved metadata; no speed simulation."""
from pathlib import Path
from fractions import Fraction
import hashlib,json
R=Path(__file__).resolve().parent;GIB=1024**3
def main():
    out=R/'hot-ffn';out.mkdir(exist_ok=True)
    census=R/'training/20261112-075449/TRAINING_COUNTS.json'
    index=R.parent/'review-verification-20261101/model-tensor-index.json'
    records={x['name']:x for x in json.loads(census.read_text())['records']}
    tensors={x['name']:x for x in json.loads(index.read_text())['tensors']}
    layouts={'IQ2_S':(256,82),'IQ3_XXS':(256,98),'IQ3_S':(256,110),'MXFP4':(32,17)}
    existing={};existing_order=[]
    for line in (R/'plans/frequency19200.plan').read_text().splitlines()[1:]:
        f=line.split();assert len(f)==6+int(f[5]);existing[f[0]]=set(map(int,f[6:]));existing_order.append(f[0])
    assert len(existing)==80
    layers=[];candidates=[];denom=0
    for layer in range(40):
        triplet=[tensors[f'blk.{layer}.ffn_{part}_exps.weight'] for part in ['gate','up','down']]
        sizes=[]
        for j,t in enumerate(triplet):
            assert t['shape']==([5120,2304,384] if j<2 else [2304,5120,384]);bs,b=layouts[t['type']]
            sizes.append(t['shape'][0]//bs*t['shape'][1]*b)
        layers.append({'layer':layer,'types':[t['type'] for t in triplet],'expert_bytes':sum(sizes),'projection_bytes':sizes,'training_observed':layer<37})
        if layer<37:
            rs=[records[t['name']] for t in triplet]
            assert all(r['counts']==rs[0]['counts'] and r['calls']==2040 and r['assignments']==12240 for r in rs)
            for r,t in zip(rs,triplet):assert r['type']==t['type_id'] and r['k']==t['shape'][0] and r['rows']==t['shape'][1] and r['experts']==384
            denom+=sum(rs[0]['counts'])
            for e,count in enumerate(rs[0]['counts']):candidates.append({'layer':layer,'expert':e,'count':count,'bytes':sum(sizes)})
    assert denom==452880
    ranked=sorted(candidates,key=lambda x:(-Fraction(x['count'],x['bytes']),-x['count'],x['bytes'],x['layer'],x['expert']))
    budgets=[0,.5,1,4,8,14,16,19];rows=[]
    for gb in budgets:
        budget=int(gb*GIB);used=0;selection=[];per_layer=[set() for _ in range(40)]
        for c in ranked:
            if c['count'] and used+c['bytes']<=budget:selection.append(c);used+=c['bytes'];per_layer[c['layer']].add(c['expert'])
        overlap=0
        for layer in range(40):
            for part in ['gate','up','down']:
                name=f'blk.{layer}.ffn_{part}_exps.weight'
                overlap+=len(existing.get(name,set())&per_layer[layer])
        reclaim=overlap*6635520
        label=str(gb).replace('.','p')+'GiB'
        selection_path=out/f'selected-{label}.json'
        selection_path.write_text(json.dumps({'budget_bytes':budget,'selected':selection,'denominator_routes':denom,'scope':'Training layers0-36 only; not speed or heldout coverage'},indent=2)+'\n')
        plan=[f'XEON_HOT_FFN_PLAN_V1 40 384 5120 2304 {len(selection)} {used}']
        for layer in range(40):
            ids=sorted(per_layer[layer]);plan.append(' '.join(map(str,[layer,*layers[layer]['types'],len(ids),*ids])))
        (out/f'hot-{label}.plan').write_text('\n'.join(plan)+'\n')
        cpu_rows=[];nslots=0
        eligible=existing_order+[t['name'] for t in tensors.values() if t['name'].startswith(('blk.37.','blk.38.','blk.39.')) and '_exps.weight' in t['name'] and t['type'] in ['IQ2_S','IQ3_XXS','IQ3_S']]
        assert len(eligible)==len(set(eligible))==87
        for name in eligible:
            t=tensors[name];layer=int(name.split('.')[1]);ids=sorted(existing.get(name,set())-per_layer[layer]);nslots+=len(ids)
            cpu_rows.append(' '.join(map(str,[name,t['type'].lower(),*t['shape'],len(ids),*ids])))
        assert nslots==19200-overlap
        (out/f'cpu40-without-hot-{label}.plan').write_text(f'XEON_IQ_CACHE_PLAN_V2 87 {nslots} {nslots*6635520}\n'+'\n'.join(cpu_rows)+'\n')
        rows.append({'budget_GiB':gb,'triplets':len(selection),'native_payload_bytes':used,'training_assignments_covered':sum(c['count'] for c in selection),'training_coverage_fraction':sum(c['count'] for c in selection)/denom,'CPU_compact_overlap_slots':overlap,'CPU_compact_reclaim_bytes':reclaim,'selection_file':str(selection_path),'new_CPU_plan_slots':nslots})
    result={'scope':'Capacity and recorded routing only; no latency simulation. No training census for37-39. RAM reclamation requires rebuilding cache, not pointer changes.',
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [census,index,R/'plans/frequency19200.plan']},
            'denominator_route_assignments':denom,'layers':layers,'budgets':rows,
            'GPU_layers37_39_native_bytes':sum(l['expert_bytes']*384 for l in layers[37:40]),
            'physical_RAM_caveat':'mlock prefix extends about16.2255GiB including0.4382GiB gaps; actualincrement depends on existing pagecache. Cache reclaimed bytes notrefilled in these plans.',
            'allocation_caveat':'Native payload only; add perbank allocatorpadding,CUDAcontext/workspaces/activations.19GiB does not fit the calibrated128k budget.'}
    (out/'CAPACITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'denominator':denom,'budgets':rows}),flush=True)
if __name__=='__main__':main()
