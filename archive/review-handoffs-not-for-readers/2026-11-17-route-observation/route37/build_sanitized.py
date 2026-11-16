"""ASAN/UBSAN for both changed CPU translation units and probe; old objects unchanged."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;D=R/'sanitized'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    D.mkdir(exist_ok=True);b=json.loads((R/'BUILD.json').read_text());flags=['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer'];commands=[];mapping={}
    for original in b['commands'][:2]:
        c=original.copy();old=c[c.index('-o')+1];new=str(D/Path(old).name);mapping[old]=new;c[c.index('-o')+1]=new;c+=flags
        subprocess.run(c,check=True,timeout=180);commands.append(c)
    c=[mapping.get(v,v) for v in b['commands'][2]]
    c=[('-Wl,--dependency-file='+str(D/'link.d')) if v.startswith('-Wl,--dependency-file=') else v for v in c]
    lib=D/'libggml-cpu.so.0.28.0';c[c.index('-o')+1]=str(lib);c+=flags
    subprocess.run(c,cwd=b['CPU_link_cwd'],check=True,timeout=120);commands.append(c)
    for name in ['libggml-cpu.so.0','libggml-cpu.so']:
        p=D/name
        if not p.exists():p.symlink_to(lib.name)
    c=json.loads((R/'PROBE_BUILD.json').read_text())['argv'];c[c.index('-o')+1]=str(D/'probe');c+=flags
    subprocess.run(c,check=True,timeout=180);commands.append(c)
    (D/'BUILD.json').write_text(json.dumps({'scope':__doc__,'commands':commands,'CPU_library_sha256':sha(lib),'probe_sha256':sha(D/'probe'),'source_sha256':{n:sha(R/n) for n in ['ggml-cpu.c','route37.cpp','route37.h','probe.cpp']},'not_instrumented':'Reused baseline objects/base library and caller integration; no TSan or model execution.'},indent=2)+'\n')
    print('Built isolated sanitizer variants')
if __name__=='__main__':main()
