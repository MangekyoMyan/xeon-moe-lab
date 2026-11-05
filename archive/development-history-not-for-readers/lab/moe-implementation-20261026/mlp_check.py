#!/usr/bin/env python3
import array,json,math,os,struct,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent; E=R.parent/'xeon-volta-engine'
O=R/('mlp-checks-io' if '--io' in sys.argv else 'mlp-checks-gateup' if '--gateup' in sys.argv else 'mlp-checks-pair' if '--pair' in sys.argv else 'mlp-checks-pinned' if '--pinned' in sys.argv else 'mlp-checks');O.mkdir(exist_ok=False)
base={k:v for k,v in os.environ.items() if not k.startswith('GGML_')}
if '--pinned' in sys.argv: base['GGML_XV_CHECK_PINNED']='1'
def run(label,env):
    path=O/(label+'.bin')
    with (O/(label+'.out')).open('w') as out,(O/(label+'.err')).open('w') as err:
        subprocess.run([str(R/'build-mlp/mlp-check'),str(path)],env={**base,**env},stdout=out,stderr=err,check=True)
    data=path.read_bytes(); offset=0;cases=[]
    while offset<len(data):
        ci,step,count=struct.unpack_from('iii',data,offset);offset+=12
        a=array.array('f');a.frombytes(data[offset:offset+count*4]);offset+=count*4
        assert all(map(math.isfinite,a));cases.append((ci,step,a))
    assert len(cases)==64
    return cases
reference=run('cpu-original',{'LD_LIBRARY_PATH':str(E/'runtime')})
results=[]
for slots in [1,3,8]:
    split=None
    for enabled in [0,1]:
        label='slots%d-mlp%d'%(slots,enabled)
        runtime=R/('runtime-io' if '--io' in sys.argv else 'runtime-gateup' if '--gateup' in sys.argv else 'runtime-pair' if '--pair' in sys.argv else 'runtime-mlp')
        env={'LD_LIBRARY_PATH':str(runtime),'GGML_XV_PLUGIN':str(runtime/'libxv-experiment.so'),
             'GGML_XV_TEST_SLOTS':str(slots),'GGML_XV_TEST_MUTABLE':'1','GGML_XV_WHOLE_MLP':str(enabled),
             'GGML_XV_MEASURE_PHASES':'1','GGML_CPU_Q8_MOE_ROWS4':'1'}
        if '--pair' in sys.argv or '--gateup' in sys.argv: env['GGML_XV_MLP_CPU_PAIR']='1'
        if '--gateup' in sys.argv: env['GGML_XV_GATE_UP_ONLY']='1'
        if '--io' in sys.argv: env['GGML_XV_PINNED_ACTIVATIONS']='1'
        actual=run(label,env); cases=[]
        if enabled==0: split=actual
        for (ci,step,a),(_,_,b),(_,_,c) in zip(actual,reference,split):
            assert len(a)==len(b)==len(c)
            rms=sum(y*y for y in b);nmse=sum((x-y)**2 for x,y in zip(a,b))/max(rms,1e-30)
            snmse=sum((x-y)**2 for x,y in zip(a,c))/max(sum(y*y for y in c),1e-30)
            assert nmse<2e-4,(label,ci,step,nmse)
            # Unsupported broadcast, GLU type, external consumers and output
            # flags must keep the original per-projection numerical path.
            if enabled and (ci>=3 or '--gateup' in sys.argv): assert a.tobytes()==c.tobytes(),(label,ci,step,'expected bitexact output differs')
            cases.append({'case':ci,'step':step,'nmse_cpu':nmse,'nmse_split':snmse,'max_abs':max(abs(x-y) for x,y in zip(a,b))})
        logs=(O/(label+'.err')).read_text().splitlines()
        stats=[json.loads(x.split(' ',1)[1]) for x in logs if x.startswith('XV_MLP ')]
        jobs=sum(x['jobs'] for x in stats)
        assert (jobs>0)==bool(enabled),(label,jobs)
        if ('--pair' in sys.argv or '--gateup' in sys.argv) and enabled:
            pairs=[json.loads(x.split(' ',1)[1]) for x in logs if x.startswith('XV_CPU_PAIR ')]
            assert sum(x['pairs'] for x in pairs)>0,(label,'pair code not reached')
        if '--pair' in sys.argv:
            assert (O/(label+'.bin')).read_bytes()==(R/'mlp-checks-pinned'/(label+'.bin')).read_bytes(),(label,'CPU pairing changed MLP output')
        if '--io' in sys.argv:
            assert (O/(label+'.bin')).read_bytes()==(R/'mlp-checks-gateup'/(label+'.bin')).read_bytes(),(label,'Pinned I/O changed output')
        results.append({'label':label,'env':env,'cases':cases,'mlp_stats':stats})
        (O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,'PASS jobs',jobs,'maxNMSE',max(x['nmse_cpu'] for x in cases),flush=True)
