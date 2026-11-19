from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;H=R/'h1'
REFERENCE_CAPTURE_ROOT=None  # Path-only relocation for offline public replay.
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(run):
    j=read(run/'RUN.json');r=read(run/'REPLAY.json');s=read(run/'SPEC.json');health=read(run/'health.json');assert j['passed'] and r['passed'] and health['status']=='exited' and health['child_returncode']==0
    expected=read(run/'PROVIDER_EXPECTED.json');assert j['providers']==expected['providers'] and j['active_backend_providers']==expected['active_backend_providers']
    assert j['bank_used_by_model'] is False and r['diagnostic_plan']==s['plan']
    assert r['bank_native_bytes']==r['bank_native_bytes_verified']==475453440 and len(r['cells'])==8
    plan=read(H/'H1_EXPECTED.json')['plans'][s['plan']]
    cap=Path(REFERENCE_CAPTURE_ROOT) if REFERENCE_CAPTURE_ROOT else Path(read(H/'CAPTURE_PIN.json')['directory'])
    sequences=['OCPS','CSOP','SPCO','POSC']*3
    membership=[]
    for line in (H/(s['plan']+'.plan')).read_text().splitlines()[1:]:
        values=line.split();membership.append(dict(layer=int(values[0]),experts=list(map(int,values[5:]))))
    assert r['plan_membership']==membership and len(membership)==40
    selected=[x['layer'] for x in membership if x['experts']]
    assert r['private_initialized_layers']==selected and r['empty_layers_bypassed']==[x['layer'] for x in membership if x['layer']<37 and not x['experts']]
    for i,c in enumerate(r['cells']):
        e=plan['all_eight_cells'][i];assert c['cell']==i and c['passed'] and c['ids']==e['global_top6'] and c['gpu_global_ids']==e['GPU_global_ids'] and c['matched_H']==e['H']
        files={m:(run/f'cell{i}-{m}.f32').read_bytes() for m in 'OCSP'}
        assert files['O']==files['C']==(cap/f'cell{i}-down.f32').read_bytes() and files['S']==files['P']
        assert len(files['O'])==5120*6*4
        for slot in range(6):
            if slot not in c['gpu_slots']:assert files['P'][slot*5120*4:(slot+1)*5120*4]==files['O'][slot*5120*4:(slot+1)*5120*4]
        first=[x for x in c['samples'] if x['first_use']];steady=[x for x in c['samples'] if not x['first_use']];assert len(first)==4 and len(steady)==48
        for block,order in enumerate(sequences):
            ss=[x for x in steady if x['block']==block];assert ''.join(x['mode'] for x in ss)==order
            for x in ss:
                assert x['output_ready_ns']>x['input_ready_ns'] and x['matched_H']==e['H']
                target=e['H'] if x['mode'] in 'SP' else 0;assert x['executed_gpu_H']==target and x['gpu_operations']==(5 if target else 0)
                if not target:assert x['join_begin_ns']==x['join_end_ns']==0
                else:assert x['join_end_ns']>=x['join_begin_ns']>0
        if i in [1,4]:
            assert e['H']==1
            for projection in c['compact']:assert projection['compact_mask'][c['gpu_slots'][0]]==(s['plan']=='H1_COMPACT')
    return r
