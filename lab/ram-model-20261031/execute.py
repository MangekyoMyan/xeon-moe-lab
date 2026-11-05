#!/usr/bin/env python3
"""Post-upgrade model comparison, preserving original evidence and deployment."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

R=Path(__file__).resolve().parent
LAB=R.parent
OLD=LAB/'strata-tc-study-20261031'
STATE=R/'STATE.json'
assert os.geteuid()==0
state={'phase':'waiting_for_preload','started':time.time(),'runs':[],
       'model':'GLM-4.5-Air Q8_0','context':131072,'power_W':150,'cache_MiB':9216,
       'model_comparison_only':True,'production_API_initially_inactive':True}
def save():STATE.write_text(json.dumps(state,indent=2)+'\n')
def run(cmd,log,timeout):
    with (R/log).open('w') as out:
        subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=timeout,check=True)
def compare(oldlabel,newlabel):
    p=subprocess.run([str(LAB/'moe-engine-20261029/tests/compare-logits'),
                      str(OLD/'results'/oldlabel/'logits.bin'),str(R/'results'/newlabel/'logits.bin')],
                     capture_output=True,text=True,check=True,timeout=60)
    j=json.loads(p.stdout)
    (R/'results'/newlabel/'historical-logit-comparison.json').write_text(json.dumps(j,indent=2)+'\n')
    # Same executable/runtime, token IDs and arithmetic paths; inspect any drift.
    assert j['all_logits_bitexact'], (newlabel,j)
    return j
save()
try:
    deadline=time.monotonic()+3700
    while True:
        assert not (R/'STOP_REQUESTED').exists(), 'Preload guard stopped; inspect evidence'
        p=LAB/'results'/R.name/'preload/run.json'
        if p.exists():
            j=json.loads(p.read_text())
            if 'returncode' in j:
                assert j['returncode']==0 and not j.get('error'),j.get('error')
                assert json.loads((R/'preload-state.json').read_text())['completed']
                break
        assert time.monotonic()<deadline,'Preload wait timeout'
        time.sleep(2)
    state['phase']='numa_validation';save()
    with (LAB/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(['numactl','--interleave=0,1','python3',str(R/'numa_model.py'),'--label','numa-before'],'numa-before.log',180)
        j=json.loads((R/'numa-before.json').read_text())
        if j['paired_sample_split_percent']<99.9:
            run(['numactl','--interleave=0,1','python3',str(R/'numa_model.py'),'--stripe','--label','numa-striped'],'numa-striped.log',900)
        state['numa_striping_performed']=j['paired_sample_split_percent']<99.9
    plan=[('long-b0','phase-v3',1024,2048,32,'english','v3-long'),
          ('long-a0','baseline',128,2048,32,'english','final-a0'),
          ('long-a1','baseline',128,2048,32,'english','final-a1'),
          ('long-b1','phase-v3',1024,2048,32,'english','v3-long'),
          ('ja1024-a','baseline',128,1024,32,'quality-ja','ja1024-a'),
          ('ja1024-b','phase-v3',1024,1024,32,'quality-ja','ja1024-b'),
          ('short-b','phase-v3',1024,64,16,'quality-ja','v3-short')]
    state['plan']=plan;state['phase']='model_comparisons';save()
    for label,mode,batch,prefix,records,language,reference in plan:
        assert not (R/'STOP_REQUESTED').exists()
        state['current']=label;save()
        cmd=['python3',str(R/'watch.py'),'--label',label,'--timeout','900','--',
             'python3',str(R/'run_model.py'),'--label',label,'--mode',mode,
             '--batch',str(batch),'--prefix',str(prefix),'--records',str(records),
             '--language',language,'--cache-mib','9216']
        run(cmd,label+'-guard.log',1000)
        j=json.loads((R/'results'/label/'result.json').read_text());assert j['success']
        assert all(x['major_faults']==0 and x['block_in']==0 for x in j['stats']['XV_PREFILL_IO'])
        previous=json.loads((OLD/'results'/reference/'result.json').read_text())
        assert j['env']==previous['env'], 'Environment drift from historical comparison'
        q=compare(reference,label)
        state['runs'].append({'label':label,'reference':reference,'prefill':j['stats']['XV_PREFILL'],
                              'timing':j['stats']['XV_TIMING'],'bitexact':q['all_logits_bitexact']});save()
        print(label,j['stats']['XV_PREFILL'],j['stats']['XV_TIMING'],flush=True)
    state['phase']='loopback_api';save()
    run(['python3',str(R/'api_smoke.py')],'api-smoke.log',1000)
    j=json.loads((R/'API_SMOKE.json').read_text());assert j['success']
    assert all(q['majflt']==0 and q['disk_read_bytes']==0 for q in j['requests'])
    state['phase']='measurements_complete';state['finished']=time.time();save()
except BaseException as e:
    state['phase']='stopped_for_inspection';state['error']=repr(e);save();raise
finally:
    owner=LAB.stat()
    for p in [R,*R.rglob('*')]:os.chown(p,owner.st_uid,owner.st_gid,follow_symlinks=False)
