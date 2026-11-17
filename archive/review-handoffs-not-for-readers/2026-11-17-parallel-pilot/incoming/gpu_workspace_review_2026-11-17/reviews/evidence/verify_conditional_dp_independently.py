#!/usr/bin/env python3
"""Independent stdlib-only check of parent's conditional 8-layer diagnostic.

Reads parent's outputs as claims; does not execute/import the parent script.
Uses sparse Pareto-frontier multiple-choice DP, up to 8 active layers, unlike
the parent's dense NumPy DP. Raw evidence path is pinned by the main audit.
"""
from collections import Counter
from functools import reduce
from math import gcd
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/scratch/1f624f9ac929')
W=ROOT/'xeon-moe-workspace-d98a17c/archive/review-handoffs-not-for-readers/2026-11-17-gpu-workspace'
P=ROOT/'audit/gpu_workspace_20261117/parent'
OUT=ROOT/'audit/gpu_workspace_20261117/evidence'
RUN=W/'pilot-capture/runs/20261117T035402Z-U1024-d22b3e04'


def load(p):
    return json.loads(p.read_bytes())


def main():
    claim=load(P/'BANK_TRADEOFF_RECOMPUTED.json')
    candidate=claim['conditional_alternative']
    expected_order=load(W/'pilot-capture/PILOT_DESIGN.json')['case_ids']
    assert claim['training_case_order']==expected_order
    assert claim['records']==11840 and claim['slots']==71040
    inv=load(RUN/'RUN.json')['native_expert_inventory']
    cost={x['layer']:x['native_triplet_bytes'] for x in inv}
    freq=Counter()
    records=[]
    for name in expected_order:
        rr=load(RUN/(name+'.json'))['T1_trace']['records']
        assert len(rr)==2368
        records+=rr
        freq.update((r['layer'],e) for r in rr for e in r['ids'])
    unit=reduce(gcd,[cost[l] for l in range(37)])
    budget=load(W/'pilot-capture/PILOT_DESIGN.json')['bank_selection_budget_bytes']
    assert unit==92160 and budget%unit==0
    B=budget//unit
    # Per exact k layers, cost -> best hit count. Dominance is safe because
    # future options are identical for states at the same (stage,k).
    states=[{} for _ in range(9)]
    states[0]={0:0}
    stage_sizes=[]
    for l in range(37):
        cu=cost[l]//unit
        best=sorted(((freq[l,e],e) for ll,e in freq if ll==l),key=lambda x:(-x[0],x[1]))
        options=[]
        total=0
        for n,(v,e) in enumerate(best[:B//cu],1):
            total+=v
            options.append((n*cu,total))
        nxt=[dict(old) for old in states]
        for k in range(8):
            for weight,value in states[k].items():
                for ow,ov in options:
                    nw=weight+ow
                    if nw>B: break
                    nv=value+ov
                    if nv>nxt[k+1].get(nw,-1):
                        nxt[k+1][nw]=nv
        for k in range(9):
            highest=-1
            frontier={}
            for weight,value in sorted(nxt[k].items()):
                if value>highest:
                    frontier[weight]=value
                    highest=value
            nxt[k]=frontier
        states=nxt
        stage_sizes.append(sum(map(len,states)))
    frontier=[]
    for k in range(1,9):
        score=max(states[k].values())
        weight=min(w for w,v in states[k].items() if v==score)
        row={'active_layer_count':k,'maximum_inclusion_count':score,'least_native_bytes_at_maximum':weight*unit}
        assert row==candidate['frontier'][k-1],(row,candidate['frontier'][k-1])
        frontier.append(row)
    target=(9*4176+9)//10
    assert target==3759
    assert all(r['maximum_inclusion_count']<target for r in frontier[:7])
    assert frontier[6]['maximum_inclusion_count']==3685
    assert frontier[7]['maximum_inclusion_count']==3796
    plan_path=P/'CONDITIONAL_CONCENTRATED_CANDIDATE.plan'
    plan=plan_path.read_text().splitlines()
    assert len(plan)==41
    assert plan[0]=='XEON_HOT_FFN_PLAN_V1 40 384 5120 2304 36 480522240'
    chosen=set()
    for l,line in enumerate(plan[1:]):
        fields=line.split()
        assert fields[:4]==[str(l),*[p['type'].upper() for p in inv[l]['projections']]]
        es=list(map(int,fields[5:]))
        assert int(fields[4])==len(es) and es==sorted(set(es))
        assert all(0<=e<384 for e in es)
        chosen.update((l,e) for e in es)
    payload=sum(cost[l] for l,e in chosen)
    hit=sum(freq[x] for x in chosen)
    H=Counter(sum((r['layer'],e) in chosen for e in r['ids']) for r in records)
    assert payload==480522240<=budget and len(chosen)==36 and len({l for l,e in chosen})==8
    assert hit==3796==sum(h*n for h,n in H.items())
    assert [H[h] for h in range(7)]==[9784,925,609,445,67,10,0]
    assert chosen=={(v['layer'],v['expert']) for v in candidate['selected']}
    assert candidate['same_exact_hits_retained'] is False
    out={'passed':True,'method':'Independent sparse Pareto-frontier DP, raw first5 only, no supplied script executed/imported',
         'target_aggregate_hit_count':target,'first8_exact_frontier':frontier,
         'min_active_layers_for_target':8,'count':36,'payload_bytes':payload,'hit_count':hit,
         'H':[H[h] for h in range(7)],'scope':'aggregate training count, not same exact hits, wall time, quality, actual GPU admission, or generalization',
         'maximum_retained_DP_states':max(stage_sizes),
         'parent_script_sha256':hashlib.sha256((P/'recompute_bank_tradeoff.py').read_bytes()).hexdigest(),
         'parent_result_sha256':hashlib.sha256((P/'BANK_TRADEOFF_RECOMPUTED.json').read_bytes()).hexdigest(),
         'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
         'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'CONDITIONAL_DP_INDEPENDENT_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':
    main()
