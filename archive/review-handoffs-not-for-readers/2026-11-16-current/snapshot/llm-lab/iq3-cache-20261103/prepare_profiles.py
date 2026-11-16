#!/usr/bin/env python3
"""Create separate opt-in config and an exact rollback snapshot; no service changes."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
config=E/'server-config.json';expected='0051756e1ce5ef563dfd25a16c8a13c275297179007745596974fb1ea4183ec1'
assert sha(config)==expected,'Live baseline changed; do not overwrite.'
raw=config.read_bytes();base=json.loads(raw);folder=R/'profiles';folder.mkdir(exist_ok=True)
old=folder/'previous.json'
if old.exists():assert old.read_bytes()==raw,'Rollback snapshot differs.'
else:old.write_bytes(raw);old.chmod(0o600)
new=copy.deepcopy(base);new['environment']['LD_LIBRARY_PATH']=str(R/'runtime')+':'+base['environment']['LD_LIBRARY_PATH']
new['environment']['GGML_CPU_IQ3_COMPACT_MIB']='65536'
new['cpu_iq3_compact_profile']={'name':'compact64','additional_ram_budget_bytes':64<<30,'minimum_mem_available_bytes':64<<30,
 'scope':'IQ3_XXS CPU MUL_MAT_ID single-token only','full_model_validation':'pending user startup',
 'selection':'whole tensors in layer/gate_up/gate/up/down order; no measured hot-expert ranking',
 'original_weights_retained':True,'runtime':str(R/'runtime')}
new['tested_on_real_model']=False
p=folder/'compact64.json';p.write_text(json.dumps(new,indent=2,ensure_ascii=False)+'\n');p.chmod(0o600)
libraries=json.loads((R/'BUILD.json').read_text())['libraries']
# All existing streaming overlays remain dependencies, including the rollback CPU/model libraries.
for name in ['libllama.so.0.8.0','libggml-cpu.so.0.28.0','libggml-base.so.0.28.0']:
 q=L/'gpu-stream-20261103/runtime'/name;libraries.append({'file':str(q),'sha256':sha(q)})
manifest={'original_config_sha256':expected,'profiles':{},'libraries':libraries,'requires_validation':str(R/'VALIDATION.json')}
for name in ['previous','compact64']:
 q=folder/(name+'.json');manifest['profiles'][name]={'file':str(q),'sha256':sha(q)}
(R/'PROFILES.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Prepared compact64 and exact previous profile; live config unchanged.')
