"""Reproduce old engine link, then compile B flag + optional metadata trace only."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;E=R/'fbc-engine';LAB=R.parent;OLD=LAB/'parallel-pilot-20261117/integrated';S=LAB/'deepseek41-bringup-20261101/source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,cwd,log):
    with log.open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=f,check=True,timeout=300)
def main():
    assert not (E/'ENGINE_BUILD.json').exists();old=json.loads((OLD/'ENGINE_BUILD.json').read_text())
    for p,h in {**old['new_objects'],**old['reused_objects']}.items():assert sha(Path(p))==h,p
    prior=old['commands'][-1];check=list(prior['argv']);control=E/'control';control.mkdir()
    for i,x in enumerate(check):
        if x.startswith('-Wl,--dependency-file='):check[i]='-Wl,--dependency-file='+str(control/'link.d')
        if x.startswith('-Wl,-Map='):check[i]='-Wl,-Map='+str(control/'link.map')
    check[check.index('-o')+1]=str(control/'libllama.so.0.8.0');run(check,prior['cwd'],control/'link.log')
    assert sha(control/'libllama.so.0.8.0')==old['libllama_sha256']=='aff89d6bd4ab7419edd6b85190ef15e7a24a810db69e10a42c331298c2e9ec1a'
    objs={};commands=[]
    for name,source,template in [('llama-context.cpp',E/'route37/llama-context.cpp',old['commands'][0]),('hot-backend.cpp',E/'hot-backend.cpp',old['commands'][2]),('fbc-trace.cpp',E/'fbc-trace.cpp',old['commands'][2])]:
        cmd=[x.replace(str(OLD),str(E)) for x in template['argv']];obj=E/(name+'.o');cmd[cmd.index('-c')+1]=str(source);cmd[cmd.index('-o')+1]=str(obj)
        cmd+=['-I'+str(S/'vendor/nlohmann')];run(cmd,template['cwd'],E/(name+'.build.log'));objs[name]=obj;commands.append(dict(argv=cmd,cwd=template['cwd']))
    cmd=list(prior['argv'])
    for i,x in enumerate(cmd):
        if x==str(OLD/'llama-context.cpp.o'):cmd[i]=str(objs['llama-context.cpp'])
        elif x==str(OLD/'hot-backend.cpp.o'):cmd[i]=str(objs['hot-backend.cpp'])
        elif x.startswith('-Wl,--dependency-file='):cmd[i]='-Wl,--dependency-file='+str(E/'link.d')
        elif x.startswith('-Wl,-Map='):cmd[i]='-Wl,-Map='+str(E/'link.map')
    cmd[cmd.index('-o')+1]=str(E/'runtime/libllama.so.0.8.0');cmd.append(str(objs['fbc-trace.cpp']));run(cmd,prior['cwd'],E/'link.log');commands.append(dict(argv=cmd,cwd=prior['cwd']))
    for name in ['libllama.so','libllama.so.0']:(E/'runtime'/name).symlink_to('libllama.so.0.8.0')
    libs=json.loads((OLD/'BUILD.json').read_text())['engine_libraries']['diagnostic']
    for name,v in libs.items():
        if name.startswith('libllama'):continue
        p=Path(v['path']);assert sha(p)==v['sha256'];stem=name.split('.so.')[0]
        for n in [name,stem+'.so',stem+'.so.0']:(E/'runtime'/n).symlink_to(p)
    receipt=dict(prior_sha256=sha(OLD/'ENGINE_BUILD.json'),control_link_byte_equal=True,commands=commands,new_objects={str(p):sha(p) for p in objs.values()},unchanged_old_objects={p:h for p,h in old['new_objects'].items() if p.endswith('llama-graph.cpp.o')},source_sha256={str(p.relative_to(E)):sha(p) for p in [E/'hot-backend.cpp',E/'hot-backend.h',E/'route37/llama-context.cpp',E/'fbc-trace.cpp',E/'fbc-trace.h']},libllama_sha256=sha(E/'runtime/libllama.so.0.8.0'),scope='same new llama allFBC; previousCPU602a/CUDA45/base2e13 unchanged; metadata trace disabled in timing')
    (E/'ENGINE_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(linked=True,libllama_sha256=receipt['libllama_sha256'])))
if __name__=='__main__':main()
