from pathlib import Path
import json,subprocess,hashlib,statistics
R=Path(__file__).resolve().parent;plan=json.loads((R/'VALIDATION_PLAN.json').read_text())
if plan['build']!=json.loads((R/'BUILD.json').read_text()):raise RuntimeError('Frozen build changed')
for job in plan['jobs']:
    name=f"validation-L{job['layer']}-{job['workload']}-C{job['chunk']}-t{job['threads']}"
    cmd=['python3',str(R/'run.py'),'--label','validation']
    for key in ['layer','workload','chunk','threads','repeats','gpu_experts']:cmd += ['--'+key.replace('_','-'),str(job[key])]
    with (R/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:print('FAILED',name,(R/(name+'.log')).read_text()[-1800:],flush=True);raise SystemExit(p.returncode)
    record=json.loads((R/'data'/name/'RESULT.json').read_text());d=record['result'];rows=d['measurements'];A=d['active_experts']
    cpu=statistics.median(v for x in rows if x['gpu_experts']==0 for v in x['wall_ms']);gpu=statistics.median(v for x in rows if x['gpu_experts']==A for v in x['wall_ms']);candidate=next(x for x in rows if x['gpu_experts']==job['gpu_experts']);b=statistics.median(candidate['wall_ms'])
    print(json.dumps({'case':name,'CPU_ms':cpu,'GPU_ms':gpu,'fixed_split_ms':b,'speed_ratio_vs_best_whole':min(cpu,gpu)/b,'gpu_experts':job['gpu_experts'],'active':A}),flush=True)
