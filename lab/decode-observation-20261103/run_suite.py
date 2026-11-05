#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import time
R=Path(__file__).resolve().parent
state={'started_unix':time.time(),'sequence':['baseline','trace','baseline','sample','baseline'],'completed':[]}
for i,mode in enumerate(state['sequence']):
    state['current']={'index':i,'mode':mode};(R/'suite-state.json').write_text(json.dumps(state,indent=2)+'\n')
    p=subprocess.run(['python3',str(R/'live_probe.py'),'--mode',mode,'--tokens','96'],capture_output=True,text=True,timeout=240)
    (R/f'suite-{i}-{mode}.stdout').write_text(p.stdout);(R/f'suite-{i}-{mode}.stderr').write_text(p.stderr)
    if p.returncode:
        state['error']=p.stderr;state['returncode']=p.returncode
        (R/'suite-state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(p.stderr,flush=True);raise SystemExit(p.returncode)
    result=json.loads(p.stdout);state['completed'].append(result);print(p.stdout.strip(),flush=True)
state['complete']=True;state['finished_unix']=time.time();(R/'suite-state.json').write_text(json.dumps(state,indent=2)+'\n')
