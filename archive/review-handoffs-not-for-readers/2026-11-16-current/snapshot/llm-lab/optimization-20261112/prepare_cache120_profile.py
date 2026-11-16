from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
build=json.loads((R/'cache120/BUILD.json').read_text())
for lib in build['libraries']:assert hashlib.sha256(Path(lib['file']).read_bytes()).hexdigest()==lib['sha256']
config=json.loads((R/'baseline-server-config.private.json').read_text())
config['environment']['LD_LIBRARY_PATH']=str(R/'cache120/runtime')+':'+config['environment']['LD_LIBRARY_PATH']
config['environment']['GGML_CPU_IQ3_COMPACT_MIB']='122880'
config['cpu_iq3_compact_profile']['name']='multi120-candidate'
config['cpu_iq3_compact_profile']['additional_ram_budget_bytes']=120*1024**3
config['cpu_iq3_compact_profile']['runtime']=str(R/'cache120/runtime')
config['cpu_iq3_compact_profile']['full_model_validation']='pending current isolated A/B'
config['study_libraries']=build['libraries']
config['optimization_study']={'name':'optimization-20261112','profile':'cache120','status':'candidate-not-validated',
 'change':'Same original arithmetic and baseline GPU layers37-39; extra compact RAM budget120GiB with64GiB reserve'}
p=R/'profiles/cache120.json';p.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n');p.chmod(0o600)
print('Prepared cache120 candidate, baseline GPU placement retained')
