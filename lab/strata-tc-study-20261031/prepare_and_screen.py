#!/usr/bin/env python3
import json,os,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
assert os.geteuid()==0
deadline=time.monotonic()+1800
while True:
    p=R/'preload-state.json'
    if p.exists() and json.loads(p.read_text()).get('completed'):break
    if time.monotonic()>deadline:raise RuntimeError('Preload did not finish')
    time.sleep(5)
assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()==''
cmd=['numactl','--interleave=0,1','python3',str(R/'numa_model.py')]
subprocess.run(cmd+['--label','numa-before'],check=True)
j=json.loads((R/'numa-before.json').read_text())
if j['paired_sample_split_percent']<99:
    subprocess.run(cmd+['--stripe','--label','numa-striped'],check=True)
    subprocess.run(cmd+['--label','numa-ready'],check=True)
else:(R/'numa-ready.json').write_text(json.dumps(j,indent=2)+'\n')
ready=json.loads((R/'numa-ready.json').read_text())
assert ready['resident_before_percent']>=99.9 and ready['paired_sample_split_percent']>=99
print('Model RAM cache and NUMA distribution ready',flush=True)
subprocess.run(['python3',str(R/'model_screen.py')],check=True)
