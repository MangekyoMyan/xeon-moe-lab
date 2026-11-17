"""Relink the existing diagnostic llama library with narrow hot-bank hooks."""
from pathlib import Path
import hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent.parent;B=LAB/'deepseek41-bringup-20261101/build';S=LAB/'deepseek41-bringup-20261101/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (R/'ENGINE_BUILD.json').exists()
    old=json.loads((LAB/'route37-integration-20261117/LINK_BUILD.json').read_text())
    for p,v in old['reused_objects'].items():assert sha(Path(p))==v['sha256'],p
    for p,h in old['new_objects'].items():assert sha(Path(p))==h,p
    for n,v in old['runtime_libraries'].items():
        p=Path(v['path']);assert sha(p)==v['sha256']
        stem=n.split('.so.')[0]
        for name in [n,stem+'.so',stem+'.so.0']:(R/'runtime'/name).symlink_to(p)
    table=json.loads((B/'compile_commands.json').read_text());commands=[];objs={}
    for name,source in [('llama-context.cpp',R/'route37/llama-context.cpp'),('llama-graph.cpp',R/'llama-graph.cpp'),('hot-backend.cpp',R/'hot-backend.cpp')]:
        entry=next(x for x in table if x['file'].endswith('/'+('llama-context.cpp' if name=='hot-backend.cpp' else name)))
        c=shlex.split(entry['command']);c[c.index('-c')+1]=str(source);obj=R/(name+'.o');c[c.index('-o')+1]=str(obj)
        c+=['-I'+str(R),'-I'+str(R/'route37'),'-I'+str(S/'ggml/src'),'-I'+str(LAB/'v100-prep-20261024/cuda-12.9.2/include'),'-std=c++17','-fno-fast-math','-ffp-contract=off']
        with (R/(name+'.build.log')).open('w') as f:subprocess.run(c,cwd=entry['directory'],stdout=f,stderr=f,check=True,timeout=240)
        commands.append(dict(argv=c,cwd=entry['directory']));objs[name]=obj
    c=old['commands'][-1].copy();reused={}
    for i,v in enumerate(c):
        if v.endswith('/route37/llama-context.cpp.o'):c[i]=str(objs['llama-context.cpp'])
        elif v=='CMakeFiles/llama.dir/llama-graph.cpp.o':c[i]=str(objs['llama-graph.cpp'])
        elif v.startswith('-Wl,--dependency-file='):c[i]='-Wl,--dependency-file='+str(R/'llama-link.d')
        elif v.startswith('-Wl,-Map='):c[i]='-Wl,-Map='+str(R/'llama-link.map')
        elif v.endswith('.o'):
            p=Path(v);p=p if p.is_absolute() else Path(old['link_cwd'])/p;reused[str(p)]=sha(p)
    lib=R/'runtime/libllama.so.0.8.0';c[c.index('-o')+1]=str(lib)
    c += [str(objs['hot-backend.cpp']),'-L'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),'-lcudart']
    with (R/'link.log').open('w') as f:subprocess.run(c,cwd=old['link_cwd'],stdout=f,stderr=f,check=True,timeout=120)
    commands.append(dict(argv=c,cwd=old['link_cwd']))
    for n in ['libllama.so','libllama.so.0']:(R/'runtime'/n).symlink_to(lib.name)
    receipt=dict(commands=commands,reused_objects=reused,new_objects={str(p):sha(p) for p in objs.values()},
        source_sha256={str(p.relative_to(R)):sha(p) for p in [R/'route37/llama-context.cpp',R/'llama-graph.cpp',R/'hot-backend.cpp',R/'hot-backend.h',R/'route37/route37.h',R/'route37/route37-bridge.h']},
        prior_link_sha256=sha(LAB/'route37-integration-20261117/LINK_BUILD.json'),libllama_sha256=sha(lib),collector='unchanged CPU provider; CLOSED ONLY in this pilot')
    (R/'ENGINE_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Isolated engine linked '+receipt['libllama_sha256'])
if __name__=='__main__':main()
