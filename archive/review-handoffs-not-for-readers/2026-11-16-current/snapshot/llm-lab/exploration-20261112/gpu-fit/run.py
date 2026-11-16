from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent.parent;ENGINE=R.parent/'deepseek41-bringup-20261101';raw=(ENGINE/'server-config.json').read_bytes();cfg=json.loads(raw)
folder=R/'gpu-fit'/('runs-'+time.strftime('%Y%m%d-%H%M%S'));folder.mkdir(mode=0o700)
env=os.environ.copy();env.update(cfg['environment']);results=[]
for cpu_layers in [37,38]:
 for batch in [1024,1536,2048]:
  assert (ENGINE/'server-config.json').read_bytes()==raw
  cmd=[str(R/'gpu-fit/estimate'),'-m',cfg['model'],'-c','131072','-t','12','-tb','24','-ngl','all','--n-cpu-moe',str(cpu_layers),'--override-tensor','per_layer_token_embd.weight=CPU','--op-offload','-fa','on','--cache-type-k','f16','--cache-type-v','f16','--lazy-mode','off','--no-repack','--fit','off','--offline','-b',str(batch),'-ub',str(batch)]
  start=time.monotonic();p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=150)
  label=f'cpu{cpu_layers}-ub{batch}';(folder/(label+'.stdout')).write_text(p.stdout);(folder/(label+'.stderr')).write_text(p.stderr);p.check_returncode()
  rows=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')];assert len(rows)==1
  item={'cpu_layers':cpu_layers,'ubatch':batch,'elapsed_s':time.monotonic()-start,'result':rows[0]};results.append(item);(folder/'RESULT.json').write_text(json.dumps({'config_sha256':hashlib.sha256(raw).hexdigest(),'cases':results,'complete':len(results)==6},indent=2)+'\n')
  print(json.dumps(item),flush=True)
print('Saved '+str(folder),flush=True)
