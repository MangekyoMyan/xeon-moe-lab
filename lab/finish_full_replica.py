import json,subprocess,sys,time
from full_replica_experiment import LAB,OUT
steps=[
 ('placement-control',[sys.executable,str(LAB/'full_replica_experiment.py'),'placement-control']),
 ('no-thp',[sys.executable,str(LAB/'without_thp.py'),sys.executable,str(LAB/'full_replica_experiment.py'),'no-thp']),
 ('locality',[sys.executable,str(LAB/'run_full_replica_counters.py')]),
 ('responses-long',[sys.executable,str(LAB/'full_replica_responses.py')]),
]
records=[]
for name,cmd in steps:
 print(json.dumps({'starting_step':name}),flush=True);start=time.monotonic()
 with (OUT/(name+'-driver.log')).open('w') as log:
  p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 records.append({'step':name,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-start})
 (OUT/'finish-index.json').write_text(json.dumps(records,indent=2))
 if p.returncode:raise RuntimeError((OUT/(name+'-driver.log')).read_text()[-4000:])
 print(json.dumps({'completed_step':name}),flush=True)
