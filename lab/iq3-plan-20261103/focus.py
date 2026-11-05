from pathlib import Path
import hashlib
import json
import subprocess
R=Path(__file__).resolve().parent
cases=json.loads((R.parent/'expert-validation-20261101/cases.json').read_text())
results=[]
for repeat in range(3):
    for direction in (['up','down'] if repeat%2==0 else ['down','up']):
        c=next(c for c in cases if c['name']=='IQ3_XXS-'+direction);name=f'focus-{direction}-{repeat}'
        command=['python3',str(R/'guard.py'),'--name',name,'--timeout','90','--','numactl','--interleave=0,1',str(R/'predecode_probe'),c['shard'],c['tensor'],'48','12']
        subprocess.run(command,check=True)
        data=[json.loads(x) for x in (R/'execution'/name/'stdout.log').read_text().splitlines() if x.startswith('{')]
        results.append({'name':name,'command':command,'rows':data})
        (R/'FOCUS.json').write_text(json.dumps({'source_sha256':hashlib.sha256((R/'predecode_probe.cpp').read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256((R/'predecode_probe').read_bytes()).hexdigest(),'results':results},indent=2)+'\n')
        print(json.dumps(next(x for x in data if x['kind']=='compact_timing')),flush=True)
