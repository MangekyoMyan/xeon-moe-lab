from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent
all_runs=[]
for name in ['selected-kv-transfer-v2','selected-kv-transfer-v2-r1','selected-kv-transfer-v2-r2']:
 folder=R/'execution'/name
 if not folder.exists():
  subprocess.run(['python3',str(R/'guard.py'),'--name',name,'--timeout','180','--','numactl','--interleave=0,1',str(R/'tiering_probe')],check=True)
 status=json.loads((folder/'status.json').read_text());assert status['exitcode']==0 and status['error'] is None
 rows=[json.loads(x) for x in (folder/'stdout.log').read_text().splitlines() if x.startswith('{')];assert len(rows)==40
 all_runs.append({'name':name,'status':status,'rows':rows})
(R/'TRANSFER_RUNS.json').write_text(json.dumps({'source_sha256':hashlib.sha256((R/'tiering_probe.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((R/'tiering_probe').read_bytes()).hexdigest(),'runs':all_runs},indent=2)+'\n')
print('Three measured runs complete; no API changes.',flush=True)
