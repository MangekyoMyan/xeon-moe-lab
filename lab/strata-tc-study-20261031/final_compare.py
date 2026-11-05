#!/usr/bin/env python3
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;assert os.geteuid()==0
assert json.loads((R/'results/operator-phase-v2/result.json').read_text())['success']
plan=[('final-b0','phase-v2',1024,2048,32,'english'),
      ('final-a0','baseline',128,2048,32,'english'),
      ('final-a1','baseline',128,2048,32,'english'),
      ('final-b1','phase-v2',1024,2048,32,'english'),
      ('quality-ja-a','baseline',128,512,32,'quality-ja'),
      ('quality-ja-b','phase-v2',1024,512,32,'quality-ja'),
      ('short-a','baseline',128,64,16,'quality-ja'),
      ('short-b','phase-v2',1024,64,16,'quality-ja')]
state=json.loads((R/'STATE.json').read_text());state.update(phase='final_baab_and_quality',final_runs=[])
(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
for label,mode,batch,prefix,records,language in plan:
    assert not (R/'STOP_REQUESTED').exists()
    cmd=['python3',str(R/'watch.py'),'--label',label,'--timeout','900','--','python3',str(R/'run_model.py'),
         '--label',label,'--mode',mode,'--batch',str(batch),'--prefix',str(prefix),'--records',str(records),'--language',language,'--cache-mib','9216']
    subprocess.run(cmd,check=True)
    result=json.loads((R/'results'/label/'result.json').read_text())
    state['final_runs'].append({'label':label,'success':result['success'],'prefill':result['stats'].get('XV_PREFILL'),'io':result['stats'].get('XV_PREFILL_IO')})
    (R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
    print('MODEL RESULT',label,result['returncode'],result['stats'].get('XV_PREFILL'),flush=True)
    if not result['success']:raise RuntimeError('Model failed: '+label)
for a,b in [('final-a0','final-b0'),('final-a1','final-b1'),('quality-ja-a','quality-ja-b'),('short-a','short-b')]:
    proc=subprocess.run([str(R.parent/'moe-engine-20261029/tests/compare-logits'),str(R/'results'/a/'logits.bin'),str(R/'results'/b/'logits.bin')],capture_output=True,text=True,check=True)
    j=json.loads(proc.stdout);(R/'results'/b/'logit-comparison.json').write_text(json.dumps(j,indent=2)+'\n');print('QUALITY',b,j,flush=True)
state['phase']='final_measurements_complete';(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
