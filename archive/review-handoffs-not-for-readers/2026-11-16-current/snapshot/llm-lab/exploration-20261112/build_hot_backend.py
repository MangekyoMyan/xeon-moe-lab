"""Rebuild the two changed llama objects plus the private backend, reusing frozen loader."""
from pathlib import Path
import fcntl,hashlib,json,shlex,subprocess
R=Path(__file__).resolve().parent;OUT=R/'hot-ffn';LAB=R.parent
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine=LAB/'deepseek41-bringup-20261101';cuda=LAB/'v100-prep-20261024/cuda-12.9.2';objdir=OUT/'objects';objdir.mkdir(exist_ok=True)
    database=json.loads((engine/'build/compile_commands.json').read_text());commands=[];objects={}
    for name in ['llama-context.cpp','llama-graph.cpp','hot-backend.cpp']:
        original='llama-graph.cpp' if name=='hot-backend.cpp' else name
        entries=[x for x in database if Path(x['file']).name==original];assert len(entries)==1;entry=entries[0]
        cmd=shlex.split(entry['command']) if 'command' in entry else entry['arguments'].copy()
        old=cmd[cmd.index('-o')+1];obj=objdir/(name+'.o')
        cmd[cmd.index('-c')+1]=str(OUT/name);cmd[cmd.index('-o')+1]=str(obj)
        cmd+=['-I'+str(OUT),'-I'+str(engine/'source/ggml/src'),'-I'+str(cuda/'include'),'-std=c++17','-fno-fast-math','-ffp-contract=off']
        subprocess.run(cmd,cwd=entry['directory'],check=True,timeout=240);commands.append({'argv':cmd,'cwd':entry['directory']})
        if name!='hot-backend.cpp':objects[old]=str(obj)
    baseline=json.loads((R/'sparse-cache/LOADER_BUILD.json').read_text());linkinfo=next(x for x in baseline['commands'] if isinstance(x,dict) and 'argv' in x)
    assert baseline['library']['sha256']=='1d1e7d722c050856ddff874b5f7b87080925555e7eb9580a7ca37fcafb2a8f26'
    runtime=OUT/'runtime';runtime.mkdir(exist_ok=True);library=runtime/'libllama.so.0.8.0'
    link=linkinfo['argv'].copy()
    for name in ['llama-context.cpp','llama-graph.cpp']:
        old='CMakeFiles/llama.dir/'+name+'.o';assert link.count(old)==1;link[link.index(old)]=str(objdir/(name+'.o'))
    link[link.index('-o')+1]=str(library.with_suffix('.new'))
    link=[('-Wl,--dependency-file='+str(OUT/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
    # Keep object references before --as-needed shared libraries.
    link.insert(link.index('CMakeFiles/llama.dir/llama.cpp.o'),str(objdir/'hot-backend.cpp.o'))
    link += ['-L'+str(cuda/'lib'),'-lcudart','-Wl,--no-undefined']
    subprocess.run(link,cwd=linkinfo['cwd'],check=True,timeout=180);library.with_suffix('.new').replace(library)
    for alias in ['libllama.so','libllama.so.0']:
        if not (runtime/alias).exists():(runtime/alias).symlink_to(library.name)
    (OUT/'ENGINE_BUILD.json').write_text(json.dumps({'compile':commands,'link':{'argv':link,'cwd':linkinfo['cwd']},'library':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'baseline_loader_manifest':str(R/'sparse-cache/LOADER_BUILD.json'),'sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['llama-context.cpp','llama-graph.cpp','hot-backend.cpp','hot-backend.h']}},indent=2)+'\n')
    print('Built isolated hot-FFN engine overlay.',flush=True)
if __name__=='__main__':main()
