#!/usr/bin/env python3
import array, hashlib, json, math, os, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parent
E=R.parent/'xeon-volta-engine'
io='--io' in sys.argv
runtime=R/('runtime-io' if io else 'runtime')
O=R/('operator-checks-io' if io else 'operator-checks'); O.mkdir(exist_ok=False)
sizes=[n for n in [8,111,1408*8,4096*8,37*3*3,37*3,37*3*3,128*4,1408*8*16,4096*8*16,128*8*128,4097*3*3,*[37*t for t in [8,9,63,64,1024]]] for _ in range(6)]
base={k:v for k,v in os.environ.items() if not k.startswith('GGML_')}
def run(label, env):
    target=O/(label+'.bin')
    with (O/(label+'.out')).open('w') as out,(O/(label+'.err')).open('w') as err:
        subprocess.run([str(R/'build/operator-check'),str(target)],env={**base,**env},stdout=out,stderr=err,check=True)
    a=array.array('f'); a.frombytes(target.read_bytes())
    assert len(a)==sum(sizes) and all(map(math.isfinite,a))
    return a
ref=run('cpu-original',{'LD_LIBRARY_PATH':str(E/'runtime')})
results=[]
variants=[('cpu-rows-'+flag,{'GGML_CPU_Q8_MOE_ROWS4':flag}) for flag in ['0','1','4096','1408']]
variants += [('gpu-axis%d-slots%d'%(axis,slots),{'GGML_XV_PLUGIN':str(runtime/'libxv-experiment.so'),
    'GGML_XV_TEST_SLOTS':str(slots),'GGML_XV_TEST_MUTABLE':'1','GGML_XV_PREFILL_TOKEN_AXIS':str(axis),
    'GGML_XV_MEASURE_PHASES':'1','GGML_CPU_Q8_MOE_ROWS4':'1'}) for axis in [0,1] for slots in [1,3,8]]
for label,env in variants:
    env['LD_LIBRARY_PATH']=str(runtime)
    if io:env['GGML_XV_PINNED_ACTIVATIONS']='1'
    actual=run(label,env); offset=0; cases=[]
    for size in sizes:
        a,b=actual[offset:offset+size],ref[offset:offset+size]; offset+=size
        mse=sum((x-y)**2 for x,y in zip(a,b))/size; rms=sum(y*y for y in b)/size
        row={'nmse':mse/max(rms,1e-30),'max_abs':max(abs(x-y) for x,y in zip(a,b))}
        assert row['nmse']<1e-4,(label,len(cases),row)
        if label.startswith('cpu'): assert a.tobytes()==b.tobytes(),(label,len(cases),row)
        cases.append(row)
    result={'label':label,'env':env,'cases':cases,'bitexact':actual.tobytes()==ref.tobytes()}
    if io:
        assert (O/(label+'.bin')).read_bytes()==(R/'operator-checks'/(label+'.bin')).read_bytes(),(label,'Pinned IO changed operator outputs')
    results.append(result); (O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(label,'PASS',len(cases),'maxNMSE',max(c['nmse'] for c in cases),'bitexact',result['bitexact'],flush=True)
(O/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.bin')},indent=2)+'\n')
