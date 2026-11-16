"""Relink pinned sparse-loader libllama; one route state stays in existing CPU DSO."""
from pathlib import Path
import hashlib,json,shlex,subprocess,shutil
R=Path(__file__).resolve().parent;LAB=R.parent;B=LAB/'deepseek41-bringup-20261101/build';S=LAB/'deepseek41-bringup-20261101/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=R/'runtime';out.mkdir(exist_ok=True);prov=R/'provenance';prov.mkdir(exist_ok=True)
    libs={'libggml-cpu.so.0.28.0':LAB/'observation-20261117/route37/runtime/libggml-cpu.so.0.28.0','libggml-cuda.so.0.28.0':LAB/'exploration-20261112/cuda-sort/release-candidate-a/libggml-cuda.so.0.28.0','libggml-base.so.0.28.0':LAB/'gpu-stream-20261103/runtime/libggml-base.so.0.28.0'}
    assert sha(libs['libggml-cpu.so.0.28.0'])=='602a6578c0784f42358fcd06607a93439fbef01ccd15482aa89506034eaf6b81'
    for name,p in libs.items():
        stem=name.split('.so.')[0]
        for link in [name,stem+'.so.0',stem+'.so']:
            q=out/link
            if not q.exists():q.symlink_to(p)
    prior=LAB/'exploration-20261112/sparse-cache/LOADER_BUILD.json';baseline=json.loads(prior.read_text())['commands'][1]
    commands=[];c=baseline['argv'].copy();c[c.index('-o')+1]=str(prov/'baseline-relink.so')
    c=[('-Wl,--dependency-file='+str(prov/'baseline-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in c]
    subprocess.run(c,cwd=baseline['cwd'],check=True,timeout=120);commands.append(c)
    baseline_sha=sha(prov/'baseline-relink.so');assert baseline_sha=='1d1e7d722c050856ddff874b5f7b87080925555e7eb9580a7ca37fcafb2a8f26',baseline_sha
    table=json.loads((B/'compile_commands.json').read_text());entry=next(x for x in table if x['file'].endswith('/llama-context.cpp'))
    original=shlex.split(entry['command']);objects=[]
    for name in ['llama-context.cpp','route37-bridge.cpp']:
        c=original.copy();c.insert(1,'-I'+str(R/'route37'));c[c.index('-c')+1]=str(R/'route37'/name);o=R/'route37'/(name+'.o');c[c.index('-o')+1]=str(o)
        c+=['-fno-fast-math','-ffp-contract=off'];subprocess.run(c,cwd=entry['directory'],check=True,timeout=180);commands.append(c);objects.append(o)
    c=[];reused={}
    for v in baseline['argv']:
        if v=='CMakeFiles/llama.dir/llama-context.cpp.o':v=str(objects[0])
        elif Path(v).name in libs:v=str(libs[Path(v).name])
        elif v.endswith('.o'):
            p=Path(v);p=p if p.is_absolute() else Path(baseline['cwd'])/p;reused[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
        if v.startswith('-Wl,--dependency-file='):v='-Wl,--dependency-file='+str(R/'llama-link.d')
        c.append(v)
    lib=out/'libllama.so.0.8.0';c[c.index('-o')+1]=str(lib)
    c+=[str(objects[1]),'-Wl,--no-undefined','-Wl,-Map='+str(prov/'llama-link.map')]
    subprocess.run(c,cwd=baseline['cwd'],check=True,timeout=120);commands.append(c)
    for name in ['libllama.so.0','libllama.so']:
        p=out/name
        if not p.exists():p.symlink_to(lib.name)
    records={}
    for label,p in [('diagnostic_cpu',libs['libggml-cpu.so.0.28.0']),('diagnostic_llama',lib)]:
        for mode,args in [('defined',['nm','-D','--defined-only']),('undefined',['nm','-D','--undefined-only']),('dynamic',['readelf','-d'])]:
            text=subprocess.check_output([*args,str(p)],text=True);(prov/(label+'-'+mode+'.txt')).write_text(text)
            records[label+'-'+mode]=text
    defined=[line.split()[-1] for line in records['diagnostic_llama-defined'].splitlines() if line.split()]
    assert not any(name.startswith('xeon_route37_') and name.endswith('_v1') for name in defined)
    for symbol in ['xeon_route37_begin_eval_v1','xeon_route37_matches_v1','xeon_route37_end_eval_v1']:
        assert records['diagnostic_llama-undefined'].count(symbol)==1 and records['diagnostic_cpu-defined'].count(symbol)==1
    model_sources={}
    for name in ['src/models/deepseek41.cpp','src/models/models.h','src/models/deepseek41.h','src/llama-graph.cpp']:
        p=S/name
        if p.exists():q=prov/'model-source'/name;q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q);model_sources[name]={'sha256':sha(p),'bytes':p.stat().st_size}
    unity=B/'src/CMakeFiles/llama.dir/Unity/unity_1_cxx.cxx';dep=unity.with_suffix(unity.suffix+'.o.d');obj=Path(str(unity)+'.o')
    for p in [unity,dep]:shutil.copy2(p,prov/p.name)
    assert 'src/models/deepseek41.cpp' in unity.read_text() and 'src/models/deepseek41.cpp' in dep.read_text()
    result={'scope':__doc__,'prior_link_sha256':sha(prior),'baseline_relink_sha256':baseline_sha,'baseline_relink_byte_identical':True,'commands':commands,'link_cwd':baseline['cwd'],'reused_objects':reused,'new_objects':{str(p):sha(p) for p in objects},'sources':{n:sha(R/'route37'/n) for n in ['llama-context.cpp','route37-bridge.cpp','route37.h']},'model_sources':model_sources,'model_Unity':{'file':str(unity),'sha256':sha(unity),'dependency_sha256':sha(dep),'object':str(obj),'object_sha256':sha(obj),'scope':'Actual include/dependency/object correspondence and preserved link inputs; not a full source rebuild.'},'diagnostic_llama_sha256':sha(lib),'runtime_libraries':{n:{'path':str(p),'sha256':sha(p)} for n,p in libs.items()},'collector_defined_only_in_CPU':True,'model_executed':False}
    (R/'LINK_BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print('Pinned baseline relink identical; diagnostic libllama linked with one collector provider')
if __name__=='__main__':main()
