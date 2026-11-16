"""Instrument changed CPU translation units and harness, preserving timing artifacts.

The reused baseline objects and base library are NOT instrumented. This detects
instrumented workspace/preparation/dot/harness failures, not every engine race.
"""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
D=R/'sanitized'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    D.mkdir(exist_ok=True)
    b=json.loads((R/'BUILD.json').read_text())
    flags=['-O1','-g','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
    replace={};commands=[]
    for original in b['commands']:
        c=original.copy();out=c[c.index('-o')+1];target=D/Path(out).name
        replace[out]=str(target);c[c.index('-o')+1]=str(target);c+=flags
        subprocess.run(c,check=True,timeout=180);commands.append(c)
    c=[replace.get(x,x) for x in b['link']['argv']]
    c=[('-Wl,--dependency-file='+str(D/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in c]
    lib=D/'libggml-cpu.so.0.28.0';c[c.index('-o')+1]=str(lib);c+=flags
    subprocess.run(c,cwd=b['link']['cwd'],check=True,timeout=180);commands.append(c)
    for name in ['libggml-cpu.so.0','libggml-cpu.so']:
        p=D/name
        if not p.exists():p.symlink_to(lib.name)
    c=json.loads((R/'PROBE_BUILD.json').read_text())['argv'];c[c.index('-o')+1]=str(D/'probe');c+=flags
    subprocess.run(c,check=True,timeout=180);commands.append(c)
    result={'scope':__doc__,'commands':commands,'link_cwd':b['link']['cwd'],'library_sha256':sha(lib),'probe_sha256':sha(D/'probe'),'source_sha256':{name:sha(R/name) for name in ['ggml-cpu.c','iq-abs.cpp','iq-abs.h','probe.cpp']}}
    (D/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Partial ASAN+UBSAN build completed')
if __name__=='__main__':main()
