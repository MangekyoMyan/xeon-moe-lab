#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

R=Path(__file__).resolve().parent;LAB=R.parent
RUNNER=LAB/'pro-review-received-20261102/deepseek-cpu-review-result/probe/run_compare.py'
PAIR=LAB/'mxfp4-target-validation-20261102/mxfp4/build-cpu-ivybridge/bin/cpu-quant-probe'
HYBRID=R/'hybrid/build-cpu-ivybridge/bin/cpu-quant-probe'
ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['correctness','performance','spread','focus'],required=True);a=ap.parse_args()
env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
provenance={}
for label,binary in [('pair',PAIR),('hybrid',HYBRID)]:
    text=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
    paths=[Path(x.split('=>',1)[1].strip().split()[0]).resolve() for x in text.splitlines() if 'libggml-' in x]
    assert all(p.parent==binary.parent.resolve() for p in paths)
    provenance[label]={'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
      'libraries':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'ldd':text}
(R/'graph-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
suites=[]
if a.phase=='correctness':
    for t,mask in [(1,'0'),(12,'0-11'),(24,'0-23')]:
        suites.append((f'boundary-t{t}',mask,['--type','all','--mode','both','--k','512','--m','259',
          '--experts','4','--topk','2','--routing','shared','--tokens','1,2,3,4,5,6,7,8,15,16,17,19',
          '--input-pad','7','--threads',str(t),'--strict-cpu','--repeats','3','--warmups','1']))
elif a.phase in ['performance','focus']:
    placements=[(1,'0'),(12,'0-11'),(24,'0-23')] if a.phase=='performance' else [(12,'0-11'),(24,'0-23')]
    for t,mask in placements:
        suites.append((f'{"real" if a.phase=="performance" else "focus"}-t{t}',mask,['--type','mxfp4','--mode','both','--real-shapes',
          '--experts','6','--topk','6','--routing','shared','--tokens',('1,2,3,4,5,6,7,8,16,17' if a.phase=='performance' else '1,2,3,4,8'),
          '--threads',str(t),'--strict-cpu','--repeats',('11' if a.phase=='performance' else '31'),'--warmups',('3' if a.phase=='performance' else '5')]))
else:
    for t,mask in [(12,'0-11'),(24,'0-23')]:
        suites.append((f'spread-t{t}',mask,['--type','mxfp4','--mode','both','--real-shapes',
          '--experts','12','--topk','6','--routing','spread','--tokens','6,8,14,17',
          '--threads',str(t),'--strict-cpu','--repeats','11','--warmups','3']))
state={'phase':a.phase,'status':'running','completed':[]};statefile=R/('phase-'+a.phase+'.json')
try:
    for name,mask,probe in suites:
        out=R/'results'/name;state['current']=name;statefile.write_text(json.dumps(state,indent=2)+'\n')
        cmd=['numactl','--interleave=0,1','python3',str(RUNNER),'--stock',str(PAIR),'--candidate',str(HYBRID),
             '--out',str(out),'--rounds',('4' if a.phase=='focus' else '2'),'--cpus',mask,'--',*probe]
        subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','600','--',*cmd],check=True)
        manifest=json.loads((out/'manifest.json').read_text());assert manifest['passed']
        rows=[json.loads(x) for x in (out/'summary.jsonl').read_text().splitlines()]
        result={'suite':name,'cases':len(rows),'all_bit_exact':True,
                'ratio_min':min(x['speed_ratio'] for x in rows),'ratio_max':max(x['speed_ratio'] for x in rows)}
        state['completed'].append(result);print(json.dumps(result),flush=True)
    state['status']='passed'
except BaseException as exc:state.update(status='failed',error=repr(exc));raise
finally:statefile.write_text(json.dumps(state,indent=2)+'\n')
