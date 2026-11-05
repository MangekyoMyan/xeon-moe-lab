#!/usr/bin/env python3
"""Validate isolated loading, registration lifetime and scheduler sizing on tiny fixtures."""
from pathlib import Path
import json
import os
import subprocess
from run_probes import check_watch
from datetime import datetime
from zoneinfo import ZoneInfo

R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
if subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode==0:
    raise SystemExit('Production API must remain stopped')
check_watch()
out=R/('validation-'+datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S'));out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
env.update(LD_LIBRARY_PATH=str(R/'runtime')+':'+str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),
           GGML_CUDA_REGISTER_HOST='1',GGML_OP_OFFLOAD_MIN_BATCH='512',GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH='1024',OMP_NUM_THREADS='1')
ldd=subprocess.check_output(['ldd',str(ENGINE/'build/bin/llama-server')],env=env,text=True)
line=next(x for x in ldd.splitlines() if 'libllama.so.0 =>' in x)
assert Path(line.split('=>',1)[1].strip().split()[0]).resolve()==R/'runtime/libllama.so.0.8.0'
(out/'ldd.txt').write_text(ldd)
commands=[('make-fixture',[str(R/'make_fixture'),str(R/'tiny-moe.gguf')]),
          ('list-devices',[str(ENGINE/'build/bin/llama-server'),'--list-devices']),
          ('loader',[str(R/'test-loader'),str(R/'tiny-moe.gguf')]),
          ('scheduler-compute',[str(R/'test-scheduler')])]
for t in [128,512,1024]:
    commands.append((f'reserve-{t}',[str(R/'reserve_probe'),str(t)]))
    commands.append((f'reserve-IQ2-{t}',[str(R/'reserve_probe'),str(t),'IQ2']))
results={}
for name,cmd in commands:
    check_watch()
    p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45)
    (out/(name+'.stdout')).write_text(p.stdout);(out/(name+'.stderr')).write_text(p.stderr)
    results[name]={'returncode':p.returncode,'command':cmd}
    if p.returncode:
        print(p.stdout[-2000:]);print(p.stderr[-4000:]);raise SystemExit('Validation failed: '+name)
    if name=='loader':assert p.stderr.count('expert DMA: registration complete')==3,p.stderr
    if name.startswith('reserve-'):results[name]['result']=json.loads(p.stdout)
    print('PASS '+name,flush=True)
env['GGML_CUDA_REGISTER_HOST']='0'
p=subprocess.run([str(R/'test-loader'),str(R/'tiny-moe.gguf')],env=env,capture_output=True,text=True,timeout=45)
(out/'loader-disabled.stdout').write_text(p.stdout);(out/'loader-disabled.stderr').write_text(p.stderr)
assert p.returncode==0 and 'expert DMA:' not in p.stderr
results['registration_disabled']={'returncode':0,'registration_calls':0}
control=R/'stock-cpu-control';control.mkdir(exist_ok=True)
alias=control/'libggml-cpu.so.0'
if not alias.exists():alias.symlink_to(ENGINE/'build/bin/libggml-cpu.so.0.28.0')
env['LD_LIBRARY_PATH']=str(control)+':'+env['LD_LIBRARY_PATH']
p=subprocess.run([str(R/'test-scheduler')],env=env,capture_output=True,text=True,timeout=45)
(out/'scheduler-stock-control.stdout').write_text(p.stdout);(out/'scheduler-stock-control.stderr').write_text(p.stderr)
assert p.returncode==0,p.stderr
optimized=[json.loads(x) for x in (out/'scheduler-compute.stdout').read_text().splitlines()]
stock=[json.loads(x) for x in p.stdout.splitlines()]
assert len(optimized)==len(stock)==12
assert all(a==b for a,b in zip(optimized,stock)), 'CPU control differs; investigate before deployment'
results['stock_cpu_control']={'returncode':0,'same_reference_errors_as_hybrid':True}
results['full_deepseek_model_loaded']=False
(out/'RESULT.json').write_text(json.dumps(results,indent=2)+'\n')
(R/'VALIDATION_LATEST.json').write_text(json.dumps({'directory':str(out),'passed':True},indent=2)+'\n')
print('Tiny fixture + existing server loader + scheduler reservation checks passed; production unchanged.')
