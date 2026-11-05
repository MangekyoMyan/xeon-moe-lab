#!/usr/bin/env python3
"""GPU-bound positive control, run LAST after all large-GLM measurements."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from watch import gpu_sample
R=Path(__file__).resolve().parent;L=R.parent;N=L/'gpu-nvfp4-20261026';B=L/'v100-prep-20261024'
manifest=json.loads((N/'loader-build.json').read_text())
for path,expected in manifest['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
model=json.loads((N/'download-state.json').read_text())['models'][0]
assert model['status']=='verified' and Path(model['path']).stat().st_size==model['bytes']
results=[]
for index,power in enumerate([150,200,200,150]):
    start=time.monotonic()
    while True:
        gpu=gpu_sample()
        if max(gpu['gpu_C'],gpu['hbm_C'])<=40:break
        assert time.monotonic()-start<600,'Cooling did not finish; stop, do not raise fans'
        print('Cooling with Standard fans',gpu['gpu_C'],gpu['hbm_C'],flush=True);time.sleep(15)
    label=f'e12-qwen-{index}-{power}w'
    command=['env','LD_LIBRARY_PATH='+':'.join(map(str,[N/'runtime',B/'build-cuda/bin',B/'cuda-12.9.2/lib'])),
        'python3',str(L/'without_thp.py'),'numactl','--physcpubind=6-11,18-23','--membind=1',
        str(B/'build-cuda/bin/llama-bench'),'-m',model['path'],'-t','6','-b','512','-ub','512',
        '--cpu-mask','0xfc0fc0','--cpu-strict','1','--poll','0','--offline','-lm','mmap','-lzm','off','--repack','0',
        '-p','512','-n','256','-d','0','-r','3','--progress','-o','json','-ngl','99','-fa','auto','-nopo','0']
    (R/'qwen-power-command.json').write_text(json.dumps({'label':label,'power_W':power,'argv':command,'model':model},indent=2)+'\n')
    subprocess.run(['python3',str(R/'watch.py'),'--label',label,'--power-limit',str(power),'--timeout','1200',
        '--env','CUDA_VISIBLE_DEVICES=GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3','--',*command],check=True)
    output=L/'results'/R.name/label
    data=json.loads((output/'stdout.txt').read_text())
    results.append({'label':label,'power_limit_W':power,'benchmark':data})
    (R/'qwen-power-results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Qwen control',label,[(x['n_prompt'],x['n_gen'],x['avg_ts']) for x in data],flush=True)
owner=R.stat()
for name in ['qwen-power-command.json','qwen-power-results.json']:os.chown(R/name,owner.st_uid,owner.st_gid)
