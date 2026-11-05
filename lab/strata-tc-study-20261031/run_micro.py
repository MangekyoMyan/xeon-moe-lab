#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--all',action='store_true');ap.add_argument('--threads',type=int,default=24);a=ap.parse_args()
out=R/'results'/a.label;out.mkdir(exist_ok=False)
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(R.parent/'prefill-study-20261030/release/prefill-v1/runtime')+':'+str(R.parent/'v100-prep-20261024/cuda-12.9.2/lib')
env['OMP_PROC_BIND']='true';env['OMP_PLACES']='threads';env['OMP_WAIT_POLICY']='PASSIVE';env['GOMP_SPINCOUNT']='0'
rows=json.loads((R/'weights/manifest.json').read_text())
if not a.all:rows=[x for x in rows if x['name'] in ['l1-e3-gate','l1-e3-down']]
report={'started':time.time(),'binary_sha256':hashlib.sha256((R/'bench').read_bytes()).hexdigest(),'cases':[]}
for x in rows:
    cmd=['numactl','--interleave=0,1',str(R/'bench'),x['path'],str(x['k']),str(x['m']),x['name'],str(a.threads)]
    with (out/(x['name']+'.jsonl')).open('w') as f,(out/(x['name']+'.stderr')).open('w') as e:
        p=subprocess.run(cmd,env=env,stdout=f,stderr=e)
    result=dict(weight=x,command=cmd,returncode=p.returncode);report['cases'].append(result)
    (out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(x['name'],'returncode',p.returncode,flush=True)
    if p.returncode:break
report['finished']=time.time();report['success']=all(x['returncode']==0 for x in report['cases'])
(out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
# Hardware guard should not confuse a numerical assertion with a hardware fault.
# Inspect run.json and individual return codes before accepting any result.
