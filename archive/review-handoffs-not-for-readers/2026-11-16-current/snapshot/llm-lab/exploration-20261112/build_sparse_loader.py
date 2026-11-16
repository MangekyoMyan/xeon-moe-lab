from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OLD=LAB/'iq-cache-next-20261103'
OUT=ROOT/'sparse-cache'
RUNTIME=OUT/'runtime'
prior=json.loads((OLD/'BUILD.json').read_text())
commands=[]
name='llama-model.cpp'
matches=[cmd for cmd in prior['commands'] if isinstance(cmd,list) and '-c' in cmd and cmd[cmd.index('-c')+1]==str(OLD/name)]
assert len(matches)==1
cmd=matches[0].copy();cmd[cmd.index('-c')+1]=str(OUT/name);cmd[cmd.index('-o')+1]=str(OUT/(name+'.o'));cmd.insert(1,'-I'+str(OUT))
commands.append(cmd);subprocess.run(cmd,check=True,timeout=240)
links=[cmd for cmd in prior['commands'] if isinstance(cmd,dict) and '-o' in cmd['argv'] and Path(cmd['argv'][cmd['argv'].index('-o')+1]).name.startswith('libllama.so.0.8.0')]
assert len(links)==1
link=links[0];argv=[str(OUT/(name+'.o')) if x==str(OLD/(name+'.o')) else x for x in link['argv']]
argv[argv.index('-o')+1]=str(RUNTIME/'libllama.so.0.8.0')
argv=[('-Wl,--dependency-file='+str(OUT/'llama-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
commands.append({'cwd':link['cwd'],'argv':argv});subprocess.run(argv,cwd=link['cwd'],check=True,timeout=120)
for alias in ['libllama.so','libllama.so.0']:
    if not (RUNTIME/alias).exists():(RUNTIME/alias).symlink_to('libllama.so.0.8.0')
base=LAB/'deepseek41-bringup-20261101/build/bin';include=LAB/'deepseek41-bringup-20261101/source/ggml/include'
cmd=['g++','-std=c++17','-O2','-I'+str(include),str(OUT/'test-plan.cpp'),'-L'+str(base),'-Wl,-rpath,'+str(base),'-lggml-base','-o',str(OUT/'test-plan')]
commands.append(cmd);subprocess.run(cmd,check=True,timeout=120)
test=subprocess.run([str(OUT/'test-plan')],capture_output=True,text=True,check=True,timeout=30)
print(test.stdout,flush=True)
(OUT/'PLAN_TEST.json').write_text(test.stdout)
record={'commands':commands,'library':{'file':str(RUNTIME/'libllama.so.0.8.0'),'sha256':hashlib.sha256((RUNTIME/'libllama.so.0.8.0').read_bytes()).hexdigest()},'sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['llama-model.cpp','model-cache.h','selection-plan.h','test-plan.cpp']}}
(OUT/'LOADER_BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
