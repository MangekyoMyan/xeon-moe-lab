#!/usr/bin/env python3
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
assert os.geteuid()==0
plan=[('screen-baseline128','baseline',128),('screen-hybrid-tc128','hybrid-tc',128),
      ('screen-stream-tc512','stream-tc',512),('screen-stream-tc1024','stream-tc',1024),
      ('screen-stream-exact1024','stream-exact',1024)]
state=json.loads((R/'STATE.json').read_text());state.update(phase='full_model_screen',model_screen=[])
(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
for label,mode,batch in plan:
    assert not (R/'STOP_REQUESTED').exists()
    cmd=['python3',str(R/'watch.py'),'--label',label,'--timeout','900','--','python3',str(R/'run_model.py'),'--label',label,'--mode',mode,'--batch',str(batch),'--prefix','2048','--records','16']
    subprocess.run(cmd,check=True)
    result=json.loads((R/'results'/label/'result.json').read_text())
    state['model_screen'].append({'label':label,'success':result['success'],'prefill':result['stats'].get('XV_PREFILL'),'io':result['stats'].get('XV_PREFILL_IO')})
    (R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    if not result['success']:raise RuntimeError('Model failed: '+label)
    if mode!='baseline':
        p=subprocess.run([str(R.parent/'moe-engine-20261029/tests/compare-logits'),str(R/'results/screen-baseline128/logits.bin'),str(R/'results'/label/'logits.bin')],capture_output=True,text=True,check=True)
        j=json.loads(p.stdout);(R/'results'/label/'logit-comparison.json').write_text(json.dumps(j,indent=2)+'\n')
        print('QUALITY',label,p.stdout.strip(),flush=True)
state['phase']='full_model_screen_complete';(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
