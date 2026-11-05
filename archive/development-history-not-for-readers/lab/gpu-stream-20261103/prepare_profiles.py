#!/usr/bin/env python3
"""Prepare reviewable configs; never change production settings or start a model."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import uuid

R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
def sha(data):return hashlib.sha256(data).hexdigest()
raw=(ENGINE/'server-config.json').read_bytes()
backup=R/'before-server-config.json'
if backup.exists():assert backup.read_bytes()==raw,'Current config changed; do not overwrite original backup'
else:backup.write_bytes(raw);backup.chmod(0o600)
base=json.loads(raw)
base.pop('cpu_profile',None)
for key in list(base['environment']):
    if key.startswith('GGML_MOE_HIST_'):del base['environment'][key]
base['environment']['LD_LIBRARY_PATH']=str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib')
overlay=json.loads((R/'OVERLAY.json').read_text())
scheduler=json.loads((R/'SCHEDULER.json').read_text())
cpu=json.loads((LAB/'mxfp4-hybrid-20261102/RUNTIME.json').read_text())['profiles']['hybrid']
assert sha(Path(cpu['library']).read_bytes())==cpu['sha256']
cpu_target=R/'runtime/libggml-cpu.so.0.28.0'
if not cpu_target.exists():
    stage=cpu_target.with_name('cpu-stage-'+uuid.uuid4().hex+'.so')
    shutil.copy2(cpu['library'],stage);stage.replace(cpu_target)
assert sha(cpu_target.read_bytes())==cpu['sha256']
for name in ['libggml-cpu.so.0','libggml-cpu.so']:
    alias=cpu_target.with_name(name)
    if not alias.exists():alias.symlink_to(cpu_target.name)
layer_bytes=json.loads((LAB/'expert-validation-20261101/expert-layer-bytes.json').read_text())
folder=R/'profiles';folder.mkdir(exist_ok=True)
manifest={'original_config_sha256':sha(raw),'profiles':{},'library':overlay['library'],
          'library_sha256':overlay['library_sha256'],'full_model_AB_passed':False}
manifest['libraries']=[{'file':overlay['library'],'sha256':overlay['library_sha256']},
 {'file':scheduler['library'],'sha256':scheduler['library_sha256']},{'file':str(cpu_target),'sha256':cpu['sha256']}]
assert sha(Path(overlay['library']).read_bytes())==overlay['library_sha256']
for name,gpu_layers,batch,stream in [('baseline',2,128,False),('gpu4-resident',4,128,False),
                                    ('gpu3-stream512',3,512,True),('gpu3-stream1024',3,1024,True)]:
    c=copy.deepcopy(base);command=c['command']
    command[command.index('--n-cpu-moe')+1]=str(40-gpu_layers)
    for flag in ['-b','-ub']:command[command.index(flag)+1]=str(batch)
    c['n_cpu_moe']=40-gpu_layers;c['gpu_resident_expert_layers']=list(range(40-gpu_layers,40))
    if stream:
        assert command.count('--no-op-offload')==1
        command[command.index('--no-op-offload')]='--op-offload'
        c['environment']['GGML_CUDA_REGISTER_HOST']='1'
        c['environment']['GGML_OP_OFFLOAD_MIN_BATCH']='512'
        c['environment']['GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH']='1024'
        c['environment']['LD_LIBRARY_PATH']=str(R/'runtime')+':'+c['environment']['LD_LIBRARY_PATH']
    c['gpu_stream_profile']={'name':name,'full_model_validation':'pending','ordinary_context':131072,
      'CPU_kernel':'hybrid4/2/1' if stream else 'original stock',
      'decode_policy':'CPU experts except resident GPU layers; IQ offload>=512, MXFP4>=1024'}
    data=(json.dumps(c,indent=2,ensure_ascii=False)+'\n').encode()
    p=folder/(name+'.json');p.write_bytes(data);p.chmod(0o600)
    static_bytes=sum(layer_bytes[str(i)] for i in range(40-gpu_layers,40))
    manifest['profiles'][name]={'file':str(p),'sha256':sha(data),'gpu_layers':gpu_layers,'batch':batch,
      'streaming':stream,'static_expert_bytes':static_bytes,'registered_expert_bytes':sum(layer_bytes[str(i)] for i in range(40-gpu_layers)) if stream else 0}
(R/'PROFILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'prepared':list(manifest['profiles']),'production_modified':False}))
