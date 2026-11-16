#!/usr/bin/env python3
"""Prepare opt-in profiles and exact rollback; never writes the active config."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
config=E/'server-config.json';expected='f9098dbb054d43f908e509fa8def3fd64298fbbf82107a34a95858b655e9ac58'
assert sha(config)==expected,'Live compact64 config changed'
raw=config.read_bytes();base=json.loads(raw);folder=R/'profiles';folder.mkdir(exist_ok=True);old=folder/'previous.json'
if old.exists():assert old.read_bytes()==raw,'Rollback changed'
else:old.write_bytes(raw);old.chmod(0o600)
libraries=json.loads((R/'BUILD.json').read_text())['libraries']
for directory,names in [(L/'iq3-cache-20261103/runtime',['libllama.so.0.8.0','libggml-cpu.so.0.28.0']),
 (L/'gpu-stream-20261103/runtime',['libllama.so.0.8.0','libggml-cpu.so.0.28.0','libggml-base.so.0.28.0'])]:
 for name in names:
  p=directory/name;libraries.append({'file':str(p),'sha256':sha(p)})
manifest={'original_config_sha256':expected,'profiles':{'previous':{'file':str(old),'sha256':sha(old)}},'libraries':libraries}
for name,mib in [('multi64',65536),('multi96',98304)]:
 new=copy.deepcopy(base);new['environment']['LD_LIBRARY_PATH']=str(R/'runtime')+':'+base['environment']['LD_LIBRARY_PATH'];new['environment']['GGML_CPU_IQ3_COMPACT_MIB']=str(mib)
 new['cpu_iq3_compact_profile']={'name':name,'additional_ram_budget_bytes':mib*1024**2,'minimum_mem_available_bytes':64<<30,
  'scope':'IQ3_XXS + IQ2_S + IQ3_S CPU MUL_MAT_ID single-token only','full_model_validation':'pending user startup',
  'selection':'IQ3_S, then other up/gate, then IQ2_S down, then IQ3_XXS down; stable layer order in ties; measured coarse priorities, no dynamic LRU',
  'original_weights_retained':True,'runtime':str(R/'runtime')};new['tested_on_real_model']=False
 p=folder/(name+'.json');p.write_text(json.dumps(new,indent=2,ensure_ascii=False)+'\n');p.chmod(0o600);manifest['profiles'][name]={'file':str(p),'sha256':sha(p)}
(R/'PROFILES.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Prepared multi64/multi96 and exact compact64 rollback; live API unchanged.')
