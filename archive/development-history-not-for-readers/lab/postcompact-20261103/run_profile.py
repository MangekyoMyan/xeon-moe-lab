from pathlib import Path
import json,subprocess,time
R=Path(__file__).resolve().parent
state={'started_unix':time.time(),'completed':[],'profile':'compact64','scope':'96-token decode, short synthetic input'}
for mode in ['baseline','trace-sample','baseline']:
 p=subprocess.run(['python3',str(R/'live_probe.py'),'--mode',mode,'--tokens','96'],capture_output=True,text=True)
 if p.returncode:
  state['error']=p.stderr;(R/'profile-state.json').write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError(p.stderr)
 row=json.loads(p.stdout);state['completed'].append(row);(R/'profile-state.json').write_text(json.dumps(state,indent=2)+'\n');print(p.stdout.strip(),flush=True)
 if mode=='trace-sample':
  f=Path(row['saved']);subprocess.run(['python3',str(R/'analyze_trace.py'),str(f)],stdout=(f/'analyze-trace.stdout').open('w'),check=True)
  p=subprocess.run(['perf','script','-i',str(f/'perf.data'),'--ns','-F','pid,tid,time,event,ip,sym,dso'],capture_output=True,text=True,check=True)
  (f/'symbols.txt').write_text(p.stdout);(f/'symbols.stderr').write_text(p.stderr)
  subprocess.run(['python3',str(R/'analyze_samples.py'),str(f)],stdout=(f/'analyze-samples.stdout').open('w'),check=True)
state['finished_unix']=time.time();state['complete']=True;(R/'profile-state.json').write_text(json.dumps(state,indent=2)+'\n')
