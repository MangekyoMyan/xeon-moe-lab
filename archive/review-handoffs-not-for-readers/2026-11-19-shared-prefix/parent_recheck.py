"""Independent reconstruction of original/candidate intervals from raw TSV."""
from pathlib import Path
import csv,json,statistics
from collections import defaultdict
def table(d,n):return list(csv.DictReader((d/f'trace169-{n}.tsv').open(),delimiter='\t'))
def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        assert b>=a
        if not merged or a>merged[-1][1]:merged.append([a,b])
        else:merged[-1][1]=max(merged[-1][1],b)
    return sum(b-a for a,b in merged)
def inspect(directory,candidate):
    ss={int(x['split']):x for x in table(directory,'splits')};tt={int(x['tensor']):x for x in table(directory,'tensors')}
    nn=defaultdict(list)
    for n in table(directory,'nodes'):nn[int(n['split'])].append(n)
    cm={int(x['correlation']):int(x['external_id']) for x in table(directory,'correlations')};kk=defaultdict(list)
    for k in table(directory,'kernels'):kk[cm[int(k['correlation'])]].append(k)
    for v in kk.values():v.sort(key=lambda k:int(k['start_ns']))
    result=[]
    for sid,cpu in ss.items():
        if cpu['backend']!='CPU':continue
        gate=[n for n in nn[sid] if tt[int(n['tensor'])]['name'].startswith('ffn_moe_gate-')]
        if not gate:continue
        assert len(gate)==1
        layer=int(tt[int(gate[0]['tensor'])]['name'].split('-')[-1]);assert layer<37
        post=ss[sid+1];assert post['backend']=='CUDA0' and post['call']==cpu['call']
        psid=sid-1 if candidate else sid+1
        pnodes=nn[psid];assert ss[psid]['backend']=='CUDA0' and ss[psid]['call']==cpu['call']
        selected=[n for n in pnodes if tt[int(n['tensor'])]['name'] in [f'ffn_gate-{layer}',f'ffn_up-{layer}',f'ffn_swiglu_limited-{layer}']]
        assert [tt[int(n['tensor'])]['name'] for n in selected]==[f'ffn_gate-{layer}',f'ffn_up-{layer}',f'ffn_swiglu_limited-{layer}']
        assert selected[0]['src1']==selected[1]['src1']
        assert selected[2]['src0']==selected[0]['tensor'] and selected[2]['src1']==selected[1]['tensor']
        assert [tt[int(n['src0'])]['type'] for n in selected[:2]]==['q6_K','q8_0']
        for n in selected:assert [int(tt[int(n['tensor'])]['ne'+str(i)]) for i in range(4)]==[2304,1,1,1]
        kernels=kk[psid]
        if candidate:
            assert len(pnodes)==3 and len(kernels)==5
            five=kernels
            assert all(tt[int(n['tensor'])]['name'] not in {f'ffn_gate-{layer}',f'ffn_up-{layer}',f'ffn_swiglu_limited-{layer}'} for n in nn[sid+1])
        else:
            stop=next(i for i,k in enumerate(kernels) if 'hc_post' in k['name'])
            clamp=[i for i,k in enumerate(kernels[:stop]) if 'swiglu_clamp_kernelIfE' in k['name']];assert len(clamp)==1
            five=kernels[clamp[0]-4:clamp[0]+1]
        assert len(five)==5 and 'quantize_q8_1' in five[0]['name'] and 'quantize_q8_1' in five[2]['name']
        assert 'ggml_type14ELi1ELb0' in five[1]['name'] and 'ggml_type8ELi1ELb0' in five[3]['name'] and 'swiglu_clamp_kernelIfE' in five[4]['name']
        assert [int(k['grid_x']) for k in (five[0],five[2],five[4])]==[20,20,9]
        assert len({(k['device'],k['context'],k['stream']) for k in five})==1
        postkernels=kk[sid+1];hc=next(i for i,k in enumerate(postkernels) if 'hc_post' in k['name'])
        last=[k for k in postkernels[:hc] if 'ggml_type14ELi1ELb1' in k['name']];assert len(last)==1,'fused down+ADD must survive'
        cs,ce=int(cpu['dispatch_ns']),int(cpu['return_ns']);intervals=[(int(k['start_ns']),int(k['end_ns'])) for k in five]
        assert cs<ce
        if not candidate:assert all(a>=ce for a,b in intervals)
        overlaps=[(max(a,cs),min(b,ce)) for a,b in intervals if max(a,cs)<min(b,ce)]
        anomalies=[max(0,a[1]-b[0]) for a,b in zip(intervals,intervals[1:])]
        result.append(dict(call=int(cpu['call']),layer=layer,prefix_busy_union_ns=union(intervals),prefix_busy_sum_ns=sum(b-a for a,b in intervals),prefix_span_ns=max(b for a,b in intervals)-min(a for a,b in intervals),cpu_ns=ce-cs,overlap_ns=union(overlaps),timestamp_overlap_ns=max(anomalies),graph=int(five[0]['graph'])))
    assert len(result)==296
    groups=defaultdict(list)
    for x in result:groups[x['call']].append(x)
    assert len(groups)==8
    for v in groups.values():assert sorted(x['layer'] for x in v)==list(range(37))
    sums=[dict(call=c,**{k:sum(x[k] for x in v) for k in ['prefix_busy_union_ns','prefix_busy_sum_ns','prefix_span_ns','overlap_ns']}) for c,v in groups.items()]
    return dict(structural_passed=True,per_layer=result,per_call=sums,median_prefix_busy_ms=statistics.median(x['prefix_busy_union_ns'] for x in sums)/1e6,median_overlap_ms=statistics.median(x['overlap_ns'] for x in sums)/1e6,anomalous_adjacent_timestamp_groups=sum(x['timestamp_overlap_ns']>0 for x in result),max_apparent_overlap_ns=max(x['timestamp_overlap_ns'] for x in result),note='Raw timestamps retained; unions avoid counting apparent same-stream timestamp overlap twice. Observer results do not establish uninstrumented speed.')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    j={mode:inspect(a.root/path,candidate) for mode,path,candidate in [('original','main/runs/observe',False),('candidate','trial/runs/observe-B',True)]}
    with a.output.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
    print(json.dumps({mode:{k:v for k,v in data.items() if k not in ['per_layer','per_call']} for mode,data in j.items()},indent=2))
