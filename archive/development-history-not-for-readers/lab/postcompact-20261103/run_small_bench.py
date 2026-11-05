from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parent
cases=json.loads((R.parent/'expert-validation-20261101/cases.json').read_text());rows=[]
for c in cases:
 if c['type'] not in ['IQ2_S','IQ3_S']:continue
 name='small-'+c['name'].lower()
 command=['python3',str(R/'guard.py'),'--name',name,'--timeout','120','--','numactl','--interleave=0,1',str(R/'iq-small-probe'),c['type'],c['shard'],c['tensor']]
 subprocess.run(command,check=True)
 data=[json.loads(x) for x in (R/'execution'/name/'stdout.log').read_text().splitlines() if x.startswith('{')]
 rows.append({'name':c['name'],'command':command,'rows':data});(R/'SMALL_BENCH.json').write_text(json.dumps(rows,indent=2)+'\n')
 for row in data:
  if row.get('kind')=='timing' and row['threads']==12:print(json.dumps(row),flush=True)
