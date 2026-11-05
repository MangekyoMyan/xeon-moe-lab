from pathlib import Path
import hashlib
import json
import subprocess
R=Path(__file__).resolve().parent
cases=json.loads((R.parent/'expert-validation-20261101/cases.json').read_text())
manifest={'source_sha256':hashlib.sha256((R/'predecode_probe.cpp').read_bytes()).hexdigest(),
          'binary_sha256':hashlib.sha256((R/'predecode_probe').read_bytes()).hexdigest(),'results':[]}
for count in [6,48]:
    for direction in ['up','down']:
        case=next(c for c in cases if c['name']=='IQ3_XXS-'+direction)
        name=f'compact-{direction}-e{count}'
        cmd=['python3',str(R/'guard.py'),'--name',name,'--timeout','90','--','numactl','--interleave=0,1',
             str(R/'predecode_probe'),case['shard'],case['tensor'],str(count)]
        subprocess.run(cmd,check=True)
        rows=[json.loads(line) for line in (R/'execution'/name/'stdout.log').read_text().splitlines() if line.startswith('{')]
        manifest['results'].append({'case':name,'rows':rows,'command':cmd})
        (R/'SCREEN.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for row in rows:
            if row.get('threads')==12:print(json.dumps(row),flush=True)
