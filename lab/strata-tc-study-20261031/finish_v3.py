#!/usr/bin/env python3
import json,os,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;assert os.geteuid()==0
def guarded(label,command):
 assert not (R/'STOP_REQUESTED').exists()
 subprocess.run(['python3',str(R/'watch.py'),'--label',label,'--timeout','900','--',*command],check=True)
guarded('operator-phase-v3',['python3',str(R/'check_operator.py'),'phase-v3','--label','operator-phase-v3','--boundaries'])
assert json.loads((R/'results/operator-phase-v3/result.json').read_text())['success']
plan=[('v3-short','phase-v3',1024,64,16,'quality-ja','short-a'),
 ('v3-ja512','phase-v3',1024,512,32,'quality-ja','quality-ja-a'),
 ('v3-long','phase-v3',1024,2048,32,'english','final-a0'),
 ('ja1024-a','baseline',128,1024,32,'quality-ja',None),
 ('ja1024-b','phase-v3',1024,1024,32,'quality-ja','ja1024-a')]
state=json.loads((R/'STATE.json').read_text());state.update(phase='v3_final_checks',v3_runs=[])
(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
def compare(a,b,name='logit-comparison.json'):
 p=subprocess.run([str(R.parent/'moe-engine-20261029/tests/compare-logits'),str(R/'results'/a/'logits.bin'),str(R/'results'/b/'logits.bin')],capture_output=True,text=True,check=True)
 j=json.loads(p.stdout);(R/'results'/b/name).write_text(json.dumps(j,indent=2)+'\n');print('QUALITY',b,j,flush=True)
for label,mode,batch,prefix,records,language,reference in plan:
 guarded(label,['python3',str(R/'run_model.py'),'--label',label,'--mode',mode,'--batch',str(batch),'--prefix',str(prefix),'--records',str(records),'--language',language,'--cache-mib','9216'])
 j=json.loads((R/'results'/label/'result.json').read_text());assert j['success'],label
 state['v3_runs'].append({'label':label,'prefill':j['stats'].get('XV_PREFILL'),'io':j['stats'].get('XV_PREFILL_IO')});(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 if reference:compare(reference,label)
 if label=='v3-long':compare('final-b0',label,'v2-equivalence.json')
# Same local API sequence on the final candidate, keeping the first evidence.
shutil.copy2(R/'API_SMOKE.json',R/'API_SMOKE_v2.json')
j=json.loads((R/'server-config.json').read_text());old=str(R/'variants/phase-v2/runtime');new=str(R/'variants/phase-v3/runtime')
j['command']=[x.replace(old,new) for x in j['command']];j['environment']={k:v.replace(old,new) for k,v in j['environment'].items()}
(R/'server-config.json').write_text(json.dumps(j,indent=2)+'\n')
subprocess.run(['python3',str(R/'api_smoke.py')],check=True)
state['phase']='v3_validation_complete';(R/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
