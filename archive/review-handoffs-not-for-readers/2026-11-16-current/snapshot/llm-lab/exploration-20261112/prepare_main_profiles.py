from pathlib import Path
import copy,hashlib,json,shutil
from heldout_probe import private_write
R=Path(__file__).resolve().parent;O=R/'cpu-coordination-mx/main-last'
def main():
    validation=json.loads((O/'VALIDATION.json').read_text());assert validation['passed']
    assert Path('/sys/bus/pci/devices/0000:84:00.0/numa_node').read_text().strip()=='1'
    source=O/'runtime/libggml-cpu.so.0.28.0';sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==validation['library_sha256']
    release=O/'release-a'
    if release.exists():assert hashlib.sha256((release/source.name).read_bytes()).hexdigest()==sha
    else:
        release.mkdir();shutil.copy2(source,release/source.name)
        for name in ['libggml-cpu.so','libggml-cpu.so.0']:(release/name).symlink_to(source.name)
    base=json.loads((R/'profiles/gpu-sort-candidate.json').read_text());assert base['optimization_study']['row_local'] is False
    assert base['environment'].get('GGML_CPU_IQ_ROW_LOCAL','0')=='0' and '--numa' not in base['command']
    assert base['command'][base['command'].index('--cpu-strict')+1]=='1' and base['command'][base['command'].index('--cpu-strict-batch')+1]=='1'
    for flag in [0,1]:
        profile=copy.deepcopy(base);name='main-last' if flag else 'main-first-control'
        profile['environment']['GGML_CPU_MAIN_LAST']=str(flag);profile['environment']['LD_LIBRARY_PATH']=str(release)+':'+profile['environment']['LD_LIBRARY_PATH']
        profile['study_libraries']=[x for x in profile['study_libraries'] if Path(x['file']).name!=source.name]+[{'file':str(release/source.name),'sha256':sha}]
        profile['cpu_iq3_compact_profile']['runtime']=str(release)
        profile['optimization_study'].update(profile=name,status='affinity-and-sanitizer-verified-awaiting-live-AB',master_last=bool(flag),gpu_pci='0000:84:00.0',gpu_NUMA_node=1,affinity_scope='Worker CPU assignments rotate; not a proof of GPU locality alone')
        target=R/'profiles'/(name+'.json')
        if target.exists():assert json.loads(target.read_text())==profile
        else:private_write(target,profile)
    print('Frozen same-binary first/last CPU assignment profiles',flush=True)
if __name__=='__main__':main()
