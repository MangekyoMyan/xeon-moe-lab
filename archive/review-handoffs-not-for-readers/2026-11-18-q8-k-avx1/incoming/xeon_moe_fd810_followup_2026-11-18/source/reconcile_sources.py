"""Read-only local/source/diagnostic-plan checks. No supplied code is executed."""
from pathlib import Path
import collections,hashlib,json,subprocess
R=Path(__file__).resolve().parent
LAB=R.parent
B=R/'incoming/xeon_moe_combined_handoff_2026-11-18'
REPO=LAB.parent/'xeon-moe-review-20261116'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan(p):
    lines=p.read_text().splitlines();h=lines[0].split();assert h[:5]==['XEON_HOT_FFN_PLAN_V1','40','384','5120','2304']
    rows=[]
    for line in lines[1:]:
        x=line.split();ids=list(map(int,x[5:]));assert len(ids)==int(x[4]) and ids==sorted(set(ids))
        rows.append(dict(layer=int(x[0]),types=x[1:4],ids=ids))
    assert [x['layer'] for x in rows]==list(range(40)) and sum(len(x['ids']) for x in rows)==int(h[5])
    return dict(rows=rows,count=int(h[5]),payload=int(h[6]),sha256=sha(p))
def main():
    v=B/'validation3_review_2026-11-18';hot=read(v/'hotpath/HOTPATH_RECOMPUTED.json')
    for name,digest in hot['source_sha256'].items():assert sha(REPO/name)==digest,name
    original=plan(LAB/'parallel-pilot-20261117/integrated/capacity.plan')
    cp=LAB/'exploration-20261112/plans/frequency19200.plan';cache={}
    for line in cp.read_text().splitlines()[1:]:
        x=line.split();ids=set(map(int,x[6:]));assert len(ids)==int(x[5]);cache[x[0]]=ids
    iq=0;matched=0;mx=0
    for row in original['rows']:
        for projection,kind in zip(['gate','up','down'],row['types']):
            for expert in row['ids']:
                if kind=='MXFP4':mx+=1
                else:
                    assert kind in ['IQ2_S','IQ3_XXS','IQ3_S'];iq+=1
                    matched+=expert in cache.get(f"blk.{row['layer']}.ffn_{projection}_exps.weight",set())
    assert (iq,matched,mx)==(95,95,10)
    cells=read(LAB/'parallel-pilot-20261117/CAPTURE_CELLS.json')['cells'];declared=read(v/'plans/H1_PLAN_VERIFICATION.json');plans={}
    for name in ['H1_COMPACT','H1_NONCOMPACT']:
        candidate=plan(v/'plans'/f'{name}.plan');assert candidate['count']==original['count'] and candidate['payload']==original['payload']
        assert candidate['sha256']==declared['plans'][name]['sha256']
        for o,c in zip(original['rows'],candidate['rows']):assert o['types']==c['types'] and len(o['ids'])==len(c['ids'])
        hits=[]
        for cell in cells:
            selected=set(candidate['rows'][cell['layer']]['ids']);ids=[x for x in cell['ordered_ids'] if x in selected]
            d=next(x for x in declared['plans'][name]['all_eight_cells'] if x['cell']==cell['cell'])
            assert ids==d['GPU_global_ids'] and len(ids)==d['H']
            hits.append(dict(cell=cell['cell'],H=len(ids),GPU_global_ids=ids))
        assert hits[1]['H']==hits[4]['H']==1
        plans[name]=dict(sha256=candidate['sha256'],payload=candidate['payload'],count=candidate['count'],hits=hits)
    frequencies=collections.Counter();routes=0
    for name in hot['source_sha256']:
        if '/pilot-capture/runs/' in name:
            j=read(REPO/name)
            for record in j['T1_trace']['records']:
                routes+=1
                for expert in record['ids']:frequencies[record['layer'],expert]+=1
    expected={(7,331):102,(7,85):2,(35,78):128,(35,100):1,(3,147):78,(7,270):77}
    assert routes==11840 and all(frequencies[k]==n for k,n in expected.items())
    cpu=LAB/'exploration-20261112/mx-prepack/release-candidate-a';build=read(cpu/'BUILD.json')
    assert sha(cpu/'sources/ggml-cpu.c')==build['sources']['ggml-cpu.c']
    dis=subprocess.check_output(['objdump','-d','--disassemble=quantize_row_q8_K',str(cpu/'libggml-cpu.so.0.28.0')],text=True)
    assert '<quantize_row_q8_K_ref@plt>' in dis and 'jmp' in dis
    (R/'CURRENT_Q8_WRAPPER.txt').write_text(dis)
    manifest=read(B/'astra62_max_trial_2026-11-18/protocol/SOURCE_MANIFEST.json')
    kernels={}
    for name in ['compact-kernel.inc','iq-small-kernels.inc']:
        p=LAB/'exploration-20261112/row-local'/name;digest=sha(p)
        matching=[x['path'] for x in manifest['files'] if x['path'].endswith('/'+name) and x['sha256']==digest]
        kernels[name]=dict(current_path=str(p),sha256=digest,matching_VM_input_files=matching)
    result=dict(source_hashes_matched=len(hot['source_sha256']),routing_records=routes,original35_IQ_projection_slots=iq,IQ_slots_in_current_CPU_plan=matched,MXFP4_projection_slots=mx,H1_plans=plans,training_frequency={f'{l}:{e}':frequencies[l,e] for l,e in expected},current_CPU_source_matches_receipt=True,current_Q8_wrapper_jumps_to_ref=True,compact_kernel_correspondence=kernels,
        scope='Static plan/source and saved route checks; live API config/library backing hashes are separately recorded. No live cache pointer lookup, model execution, patch application or build.')
    (R/'SOURCE_RECONCILIATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
