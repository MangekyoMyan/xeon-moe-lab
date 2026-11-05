from pathlib import Path
import hashlib,json,shlex,subprocess,time
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';B=E/'build/src'
if hashlib.sha256((E/'source/src/llama-context.cpp').read_bytes()).hexdigest()!=json.loads((R/'CAPTURE_SOURCE.json').read_text())['original_sha256']:raise RuntimeError('Frozen source changed')
flags={}
for line in (B/'CMakeFiles/llama.dir/flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        k,v=line.split(' = ',1);flags[k]=shlex.split(v)
compile=['g++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(R),'-I'+str(E/'source/src'),'-c',str(R/'llama-context-shadow.cpp'),'-o',str(R/'llama-context-shadow.o')]
start=time.monotonic();p=subprocess.run(compile,capture_output=True,text=True,timeout=120)
(R/'capture-build.log').write_text(p.stdout+p.stderr);print('compile',p.returncode,round(time.monotonic()-start,2),p.stderr[-1500:],flush=True)
if p.returncode:raise SystemExit(p.returncode)
base=json.loads((L/'iq-cache-next-20261103/BUILD.json').read_text())
linkinfo=next(c for c in base['commands'] if isinstance(c,dict) and c['cwd']==str(B))
link=list(linkinfo['argv']);old='CMakeFiles/llama.dir/llama-context.cpp.o';assert link.count(old)==1;link[link.index(old)]=str(R/'llama-context-shadow.o')
out=R/'runtime';out.mkdir(exist_ok=True);target=out/'libllama.so.0.8.0';link[link.index('-o')+1]=str(target)
link=[('-Wl,--dependency-file='+str(R/'capture-link.d')) if a.startswith('-Wl,--dependency-file=') else a for a in link]
p=subprocess.run(link,cwd=B,capture_output=True,text=True,timeout=60);(R/'capture-link.log').write_text(p.stdout+p.stderr)
if p.returncode:raise RuntimeError(p.stderr[-1500:])
for name in ['libllama.so','libllama.so.0']:
    if not (out/name).exists():(out/name).symlink_to(target.name)
(R/'CAPTURE_BUILD.json').write_text(json.dumps({'compile':compile,'link':link,'link_cwd':str(B),'library':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'helper_sha256':hashlib.sha256((R/'shadow_capture.h').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((R/'llama-context-shadow.cpp').read_bytes()).hexdigest(),
    'preserved_model_object':str(L/'iq-cache-next-20261103/llama-model.cpp.o'),'production_modified':False},indent=2)+'\n')
print('Observer overlay linked; production is untouched')
