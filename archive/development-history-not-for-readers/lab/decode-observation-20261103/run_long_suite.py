#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
R=Path(__file__).resolve().parent
state={'context_tokens':3072,'generation_tokens':64,'completed':[]}
for mode in ['baseline','trace','baseline']:
    p=subprocess.run(['python3',str(R/'live_probe.py'),'--mode',mode,'--tokens','64','--context-tokens','3072'],capture_output=True,text=True,timeout=240)
    if p.returncode:
        state['error']=p.stderr;(R/'long-suite-state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(p.stderr,flush=True);raise SystemExit(p.returncode)
    row=json.loads(p.stdout);state['completed'].append(row);(R/'long-suite-state.json').write_text(json.dumps(state,indent=2)+'\n');print(p.stdout.strip(),flush=True)
state['complete']=True;(R/'long-suite-state.json').write_text(json.dumps(state,indent=2)+'\n')
