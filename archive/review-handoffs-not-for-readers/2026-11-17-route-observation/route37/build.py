"""Compile a diagnostic CPU library and syntax-check caller integration only."""
from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parent;LAB=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    b=json.loads((LAB/'exploration-20261112/mx-prepack/release-candidate-a/BUILD.json').read_text())
    commands=[]
    c=b['cpp'].copy();c.insert(1,'-I'+str(R));c[c.index('-c')+1]=str(R/'route37.cpp');c[c.index('-o')+1]=str(R/'route37.o')
    subprocess.run(c,check=True,timeout=180);commands.append(c)
    original=b['variants']['candidate'];c=original['compile'].copy();old=c[c.index('-o')+1]
    c.insert(1,'-I'+str(R));c[c.index('-c')+1]=str(R/'ggml-cpu.c');c[c.index('-o')+1]=str(R/'ggml-cpu.c.o')
    subprocess.run(c,check=True,timeout=180);commands.append(c)
    runtime=R/'runtime';runtime.mkdir(exist_ok=True);lib=runtime/'libggml-cpu.so.0.28.0'
    c=[str(R/'ggml-cpu.c.o') if v==old else v for v in original['link']['argv']]
    c=[('-Wl,--dependency-file='+str(R/'link.d')) if v.startswith('-Wl,--dependency-file=') else v for v in c]
    c[c.index('-o')+1]=str(lib);c+=[str(R/'route37.o'),'-Wl,--no-undefined']
    subprocess.run(c,cwd=original['link']['cwd'],check=True,timeout=120);commands.append(c)
    for n in ['libggml-cpu.so','libggml-cpu.so.0']:
        p=runtime/n
        if not p.exists():p.symlink_to(lib.name)
    table=json.loads((LAB/'deepseek41-bringup-20261101/build/compile_commands.json').read_text())
    entry=next(x for x in table if x['file'].endswith('/llama-context.cpp'));c=shlex.split(entry['command'])
    oi=c.index('-o');del c[oi:oi+2];ci=c.index('-c');del c[ci:ci+2]
    c+=['-I'+str(R),'-fsyntax-only']
    for name in ['llama-context.cpp','route37-bridge.cpp']:
        cmd=c+[str(R/name)];subprocess.run(cmd,cwd=entry['directory'],check=True,timeout=120);commands.append(cmd)
    result={'scope':__doc__,'commands':commands,'CPU_link_cwd':original['link']['cwd'],'source_sha256':{n:sha(R/n) for n in ['route37.h','route37.cpp','route37-bridge.h','route37-bridge.cpp','ggml-cpu.c','llama-context.cpp']},'CPU_library_sha256':sha(lib),'libllama_linked':False,'full_model_executed':False}
    (R/'BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print('Diagnostic CPU linked; caller source syntax checked; full libllama not linked')
if __name__=='__main__':main()
