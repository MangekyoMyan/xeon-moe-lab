import json,subprocess,sys
from full_replica_experiment import LAB,OUT
records=[]
for stage in ['pages-large','pages-fine']:
 cmd=[sys.executable,str(LAB/'full_replica_experiment.py'),stage]
 if stage=='pages-fine':cmd=[sys.executable,str(LAB/'without_thp.py')]+cmd
 print(json.dumps({'page_query':stage}),flush=True)
 with (OUT/(stage+'-driver.log')).open('w') as log:
  p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 records.append({'stage':stage,'command':cmd,'exit_code':p.returncode})
 (OUT/'page-query-index.json').write_text(json.dumps(records,indent=2))
 if p.returncode:raise RuntimeError((OUT/(stage+'-driver.log')).read_text()[-3000:])
print('Page queries complete',flush=True)
