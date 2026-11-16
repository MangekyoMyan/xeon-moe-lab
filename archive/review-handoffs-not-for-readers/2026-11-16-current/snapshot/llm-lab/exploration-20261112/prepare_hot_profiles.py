"""Freeze a validated engine and separate disabled/all-miss/small/full residency profiles."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    validation=json.loads((OUT/'BACKEND_VALIDATION.json').read_text());assert validation['passed']
    build=json.loads((OUT/'ENGINE_BUILD.json').read_text());lib=Path(build['library']);assert sha(lib)==build['sha256']==validation['library_sha256']
    for name,value in build['sources'].items():assert sha(OUT/name)==value
    release=OUT/'release-a'
    if release.exists():assert sha(release/lib.name)==sha(lib)
    else:
        release.mkdir();shutil.copy2(lib,release/lib.name)
        for name in ['libllama.so','libllama.so.0']:(release/name).symlink_to(lib.name)
    rows=[]
    for profile,enabled,budget,cpu_layers in [('hot-disabled',0,'0GiB',37),('hot-miss',1,'0GiB',37),('hot-small',1,'0p5GiB',37),('hot14',1,'14GiB',40),('hot16',1,'16GiB',40)]:
        hot_plan=OUT/f'hot-{budget}.plan';assert hot_plan.is_file()
        c=json.loads((R/'profiles/final.json').read_text());env=c['environment'];env['LD_LIBRARY_PATH']=str(release)+':'+env['LD_LIBRARY_PATH']
        env['GGML_XEON_HOT_FFN']=str(enabled);env['GGML_XEON_HOT_PLAN']=str(hot_plan)
        c['study_libraries']=[x for x in c['study_libraries'] if Path(x['file']).name!=lib.name]+[{'file':str(release/lib.name),'sha256':sha(lib)}]
        c['study_files'].append({'file':str(hot_plan),'sha256':sha(hot_plan)})
        if cpu_layers==40:
            cpu_plan=OUT/f'cpu40-without-hot-{budget}.plan';assert cpu_plan.is_file()
            c['command'][c['command'].index('--n-cpu-moe')+1]='40';c['n_cpu_moe']=40;c['gpu_resident_expert_layers']=[]
            old=env['GGML_CPU_IQ_PLAN'];env['GGML_CPU_IQ_PLAN']=str(cpu_plan)
            c['study_files']=[x for x in c['study_files'] if x['file']!=old]+[{'file':str(cpu_plan),'sha256':sha(cpu_plan)}]
            c['cpu_iq3_compact_profile'].update(name='frequency19200-minus-GPU-hot',selection='Remove GPU hot overlap; add zero-selected inventory for formerly GPU37-39; no refill',compact_payload_bytes=int(cpu_plan.read_text().splitlines()[0].split()[3]))
        c['tested_on_real_model']=False
        c['optimization_study'].update(profile=profile,status='isolated-scheduler-validated-awaiting-fullmodel',hot_backend_enabled=bool(enabled),hot_plan=str(hot_plan),GPU_hot_payload_bytes=int(hot_plan.read_text().splitlines()[0].split()[-1]),CPU_expert_layers=cpu_layers,
            GPU_CPU_precision_note='GPU activation quantization/reduction may differ; standalone numeric checks are not wholemodel quality proof')
        if enabled:c['gpu_stream_profile']['decode_policy']='Dedicated FFN: GPU resident hits concurrently with CPU misses; T>1 existing graph/offload thresholds'
        target=R/'profiles'/f'{profile}.json';raw=(json.dumps(c,indent=2)+'\n').encode()
        if target.exists():assert target.read_bytes()==raw
        else:target.write_bytes(raw);target.chmod(0o600)
        rows.append({'profile':profile,'sha256':sha(target)})
    (OUT/'PROFILES.json').write_text(json.dumps({'library_sha256':sha(lib),'profiles':rows,'not_deployed':True},indent=2)+'\n')
    print(json.dumps({'frozen_engine':str(release),'profiles':[r['profile'] for r in rows]}),flush=True)
if __name__=='__main__':main()
