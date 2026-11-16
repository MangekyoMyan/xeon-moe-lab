"""Freeze tested coordination overlay files; never relink a mapped release."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',choices=['mx','iq'],required=True);a=p.parse_args()
    group=R/('cpu-coordination-'+a.base);release=group/'release-combined-a'
    source=group/'combined/runtime/libggml-cpu.so.0.28.0'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    validation=json.loads((group/'VALIDATION.json').read_text())
    assert validation['passed'] and validation['library_sha256']==sha
    if release.exists():
        assert hashlib.sha256((release/source.name).read_bytes()).hexdigest()==sha
    else:
        release.mkdir();shutil.copy2(source,release/source.name)
        for name in ['libggml-cpu.so','libggml-cpu.so.0']:(release/name).symlink_to(source.name)
    base_name='iq-rhs-candidate' if a.base=='iq' else 'gpu-sort-candidate'
    profile=json.loads((R/'profiles'/(base_name+'.json')).read_text());name='coord-'+a.base+'-combined'
    env=profile['environment'];env['LD_LIBRARY_PATH']=str(release)+':'+env['LD_LIBRARY_PATH']
    profile['study_libraries']=[x for x in profile['study_libraries'] if Path(x['file']).name!=source.name]+[{'file':str(release/source.name),'sha256':sha}]
    profile['optimization_study'].update(profile=name,status='graph-sanitizer-verified-awaiting-live-AB',coordination=['active-expert-relaxed-counters','OpenMP-final-barrier','MX-worker-safety-mask'])
    profile['cpu_iq3_compact_profile']['runtime']=str(release)
    target=R/'profiles'/(name+'.json');data=(json.dumps(profile,indent=2)+'\n').encode()
    if target.exists() and target.read_bytes()!=data:raise RuntimeError('Refusing to replace a frozen coordination profile')
    target.write_bytes(data);target.chmod(0o600)
    (release/'MANIFEST.json').write_text(json.dumps({'source_library':str(source),'sha256':sha,'profile':name,'base_profile':base_name},indent=2)+'\n')
    print(name)
if __name__=='__main__':main()
