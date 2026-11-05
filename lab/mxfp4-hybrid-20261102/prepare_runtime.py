#!/usr/bin/env python3
"""Freeze CPU-only overrides and check original server loader/device enumeration."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
raw=(ENGINE/'server-config.json').read_bytes();config=json.loads(raw)
expected='ffec21d7555203eea4331e375b5da5132107eaa9d3edaed427d2dcf4bebb559e'
assert hashlib.sha256(raw).hexdigest()==expected,'Current config changed; review before preparing profiles.'
backup=R/'baseline-server-config.json';backup.write_bytes(raw);backup.chmod(0o600)
manifest={'baseline_config_sha256':expected,'engine':str(ENGINE),'profiles':{},'loader_checks':{}}
paths={'pair':LAB/'mxfp4-target-validation-20261102/mxfp4/build-cpu-ivybridge/bin/libggml-cpu.so.0.28.0',
       'hybrid':R/'hybrid/build-cpu-ivybridge/bin/libggml-cpu.so.0.28.0',
       'stock-hist':R/'profile-stock/libggml-cpu.so.0.28.0'}
for name,source in paths.items():
    folder=R/'runtime'/name;folder.mkdir(parents=True,exist_ok=False)
    target=folder/source.name;shutil.copy2(source,target);target.chmod(0o444)
    for alias in ['libggml-cpu.so.0','libggml-cpu.so']:(folder/alias).symlink_to(target.name)
    env={k:v for k,v in os.environ.items() if not k.startswith(('LLAMA_','GGML_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']}
    env.update(config['environment']);env['LD_LIBRARY_PATH']=str(folder)+':'+config['environment']['LD_LIBRARY_PATH']
    ldd=subprocess.check_output(['ldd',str(ENGINE/'build/bin/llama-server')],env=env,text=True)
    cpu_line=next(x for x in ldd.splitlines() if 'libggml-cpu.so.0 =>' in x)
    resolved=Path(cpu_line.split('=>',1)[1].strip().split()[0]).resolve();assert resolved==target.resolve()
    base_line=next(x for x in ldd.splitlines() if 'libggml-base.so.0 =>' in x)
    assert Path(base_line.split('=>',1)[1].strip().split()[0]).resolve().parent==ENGINE/'build/bin'
    p=subprocess.run([str(ENGINE/'build/bin/llama-server'),'--list-devices'],env=env,capture_output=True,text=True,timeout=30)
    (folder/'list-devices.stdout').write_text(p.stdout);(folder/'list-devices.stderr').write_text(p.stderr)
    assert p.returncode==0 and 'CUDA0' in p.stdout+p.stderr,(name,p.returncode)
    manifest['profiles'][name]={'directory':str(folder),'library':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    manifest['loader_checks'][name]={'correct_CPU_DSO':True,'original_base_DSO':True,'CUDA0_enumerated':True,'model_loaded':False,'ldd':ldd}
(R/'RUNTIME.json').write_text(json.dumps(manifest,indent=2)+'\n')
(R/'captures').mkdir(exist_ok=True)
print(json.dumps({'profiles':list(paths),'original_server_loader_and_CUDA_device_checks':'passed','API_unchanged':True}),flush=True)
