from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parent;L=R.parent;guard=L/'iq3-plan-20261103/guard.py'
shard=L/'models/orcarouter--DeepSeek-V4.1-Flash-1017-abliterated-GGUF/Mix-256G/DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00003-of-00007.gguf'
records=[]
for rep in range(3):
 for direction,tensor in [('up','blk.4.ffn_up_exps.weight'),('down','blk.0.ffn_down_exps.weight')]:
  name=f'cache-rotating-{direction}-{rep}'
  cmd=['python3',str(guard),'--name',name,'--timeout','90','--','numactl','--interleave=0,1',str(R/'test-cache'),str(shard),tensor]
  subprocess.run(cmd,check=True)
  folder=guard.parent/'execution'/name;rows=[json.loads(x) for x in (folder/'stdout.log').read_text().splitlines() if x.startswith('{')]
  records.append({'name':name,'direction':direction,'command':cmd,'rows':rows,'status':json.loads((folder/'status.json').read_text())})
  (R/'REAL_GRAPH.json').write_text(json.dumps(records,indent=2)+'\n')
print('All rotating-expert graph comparisons passed',flush=True)
