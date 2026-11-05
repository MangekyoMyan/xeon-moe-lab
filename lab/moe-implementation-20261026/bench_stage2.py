#!/usr/bin/env python3
import fcntl,json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
with (R.parent/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
assert len(json.loads((R/'mlp-checks-pair/results.json').read_text()))==6
results=[]
def run(label,opts,power=150):
    assert not (R/'STOP_REQUESTED').exists()
    cmd=['python3',str(R/'watch.py'),'--label',label,'--power-limit',str(power),'--timeout','900','--',
         'python3',str(R/'run_probe.py'),label,'--interleave','--records','256','--measure',*opts]
    print('Starting',label,flush=True);subprocess.run(cmd,check=True)
    x=json.loads((R/'model-runs'/label/'result.json').read_text())
    if '--pair' in opts:assert sum(v['pairs'] for v in x['XV_CPU_PAIR'])>10000
    if '--mlp' in opts or '--pair' in opts:assert sum(v['jobs'] for v in x['XV_MLP'])>10000
    results.append({'label':label,'power_W':power,'result':x})
    (R/'stage2-summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(label,x['XV_TIMING'],x['XV_CPU_PAIR'],flush=True)
    with (R/'JOURNAL.jsonl').open('a') as f:f.write(json.dumps({'event':'stage2_model_run','label':label,'result':x})+'\n')
run('e05-pair-quality',['--pair','--save-logits'])
assert (R/'model-runs/e05-pair-quality/logits.bin').read_bytes()==(R/'model-runs/e03-mlp-hostfix-quality/logits.bin').read_bytes(),'CPU input sharing changed model logits'
# Forward/reverse blocks. Same fixed tokens, page placement and process policy.
variants=[('base',[]),('rows-all',['--rows','1']),('rows-gate',['--rows','4096']),('mlp',['--mlp']),('pair',['--pair']),('pair-rows',['--pair','--rows','1'])]
for block,order in enumerate([variants,list(reversed(variants))]):
    for name,opts in order:run('e06-b%d-%s'%(block,name),opts)
print('Stage2 complete',flush=True)
owner=R.stat();os.chown(R/'stage2-summary.json',owner.st_uid,owner.st_gid)
