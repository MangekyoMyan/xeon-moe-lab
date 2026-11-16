from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;N=R/'native-pool'
def main():
    result=json.loads((N/'VALIDATION.json').read_text());assert result['passed']
    source=N/'normal/runtime/libggml-cpu.so.0.28.0';sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==result['library_sha256']
    release=N/'release-a'
    if release.exists():assert hashlib.sha256((release/source.name).read_bytes()).hexdigest()==sha
    else:
        release.mkdir();shutil.copy2(source,release/source.name)
        for name in ['libggml-cpu.so','libggml-cpu.so.0']:(release/name).symlink_to(source.name)
    profile=json.loads((R/'profiles/gpu-sort-candidate.json').read_text());env=profile['environment']
    assert profile['optimization_study']['row_local'] is False
    assert not any('ROW_LOCAL' in key for key in env)
    assert profile['command'][profile['command'].index('--poll')+1]=='0'
    assert profile['command'][profile['command'].index('--poll-batch')+1]=='0'
    env['LD_LIBRARY_PATH']=str(release)+':'+env['LD_LIBRARY_PATH']
    profile['study_libraries']=[x for x in profile['study_libraries'] if Path(x['file']).name!=source.name]+[{'file':str(release/source.name),'sha256':sha}]
    profile['cpu_iq3_compact_profile']['runtime']=str(release)
    profile['optimization_study'].update(profile='native-pool',status='graph-lifecycle-sanitizer-verified-awaiting-live-AB',CPU_graph_runtime='native pthread; OpenMP retained for cache and quantizer initialization',native_master_affinity='last CPU in pool mask; no row-local assumption')
    target=R/'profiles/native-pool.json';data=(json.dumps(profile,indent=2)+'\n').encode()
    if target.exists():assert target.read_bytes()==data
    else:target.write_bytes(data);target.chmod(0o600)
    (release/'MANIFEST.json').write_text(json.dumps({'source':str(source),'sha256':sha,'profile':'native-pool'},indent=2)+'\n')
    print('Frozen native-pool profile',flush=True)
if __name__=='__main__':main()
