#!/usr/bin/env python3
"""Fixed-token probe with full logits and actual synchronized prompt timings."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;LAB=R.parent
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True)
ap.add_argument('--mode',choices=['baseline','hybrid-tc','stream-exact','stream-tc','phase-tc','phase-v2','phase-v3'],required=True)
ap.add_argument('--batch',type=int,default=128);ap.add_argument('--prefix',type=int,default=2048)
ap.add_argument('--records',type=int,default=16);ap.add_argument('--language',choices=['english','japanese','quality-ja'],default='english');ap.add_argument('--cache-mib',type=int,choices=[9216,10240],default=10240);a=ap.parse_args()
out=R/'results'/a.label;out.mkdir(exist_ok=False)
base=json.loads((LAB/'api-server-20261030/server-config.json').read_text())
original=LAB/'prefill-study-20261030/release/prefill-v1/runtime'
rt=original if a.mode=='baseline' else R/'variants'/('stream-tc' if a.mode.startswith('stream-') else a.mode)/'runtime'
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
env.update({k:v.replace(str(original),str(rt)) for k,v in base['environment'].items()})
env.update(GGML_XV_PROBE_BATCH=str(a.batch),GGML_XV_PROBE_THREADS='12',GGML_XV_PROBE_BATCH_THREADS='24',GGML_XV_MEASURE_PHASES='1',GOMP_SPINCOUNT='0')
env['LD_PRELOAD']=str(rt/'libxv-auto-step.so')
if a.mode!='baseline':env['GGML_XV_TC_PREFILL']='0' if a.mode=='stream-exact' else '1'
if a.mode in ['phase-tc','phase-v2','phase-v3']:env.update(GGML_XV_STREAM_ACTIVE='1',GGML_XV_STREAM_PREFILL_ONLY='1',GGML_XV_STREAM_MIB='256')
if a.cache_mib==9216:env.update(GGML_XV_CACHE_MIB='9216',GGML_XV_PLAN=str(R/'glm-air-q8-9g.txt'))
if a.mode.startswith('stream-'):
    for k in list(env):
        if k.startswith('GGML_XV_ADAPTIVE') or k in ['GGML_XV_AUTO_STEP','GGML_XV_DEFER_COMMIT','LD_PRELOAD']:env.pop(k)
    env.update(GGML_XV_STREAM_ACTIVE='1',GGML_XV_STREAM_MIB='256',GGML_XV_WHOLE_MLP='0',GGML_XV_GATE_UP_ONLY='0',GGML_XV_MLP_CPU_PAIR='0')
cmd=['python3',str(LAB/'moe-implementation-20261026/release/without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',str(R/'model-probe'),base['model'],str(out/'logits.bin'),str(R/(a.language+'.txt')),str(a.prefix),str(a.records)]
record={'args':vars(a),'command':cmd,'env':{k:v for k,v in env.items() if k.startswith(('GGML_','OMP_','GOMP_')) or k in ['LD_LIBRARY_PATH','LD_PRELOAD']},'started':time.time(),'binary_sha256':hashlib.sha256((R/'model-probe').read_bytes()).hexdigest()}
with (out/'stdout.txt').open('w') as f,(out/'stderr.txt').open('w') as e:
    p=subprocess.Popen(cmd,env=env,stdout=f,stderr=e);record['model_pid']=p.pid
    (out/'invocation.json').write_text(json.dumps(record,indent=2)+'\n');rc=p.wait()
record['returncode']=rc;record['finished']=time.time();record['stats']={}
for line in (out/'stderr.txt').read_text(errors='replace').splitlines():
    if line.startswith('XV_') and ' {' in line:
        key,raw=line.split(' ',1)
        try:record['stats'].setdefault(key,[]).append(json.loads(raw))
        except ValueError:pass
u=resource.getrusage(resource.RUSAGE_CHILDREN);record['resources']={'user_s':u.ru_utime,'system_s':u.ru_stime,'major_faults':u.ru_majflt,'minor_faults':u.ru_minflt,'block_in':u.ru_inblock}
record['success']=rc==0
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(a.label,'returncode',rc,'prefill',record['stats'].get('XV_PREFILL'),flush=True)
