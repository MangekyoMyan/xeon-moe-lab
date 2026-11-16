from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'sparse-cache'
OUT=ROOT/'row-local'
RUNTIME=OUT/'runtime';RUNTIME.mkdir(exist_ok=True)
prior=json.loads((BASE/'CPU_BUILD.json').read_text())
compile_cmd=next(cmd for cmd in prior['commands'] if isinstance(cmd,list) and '-c' in cmd and cmd[cmd.index('-c')+1]==str(BASE/'iq3-cache.cpp')).copy()
compile_cmd[compile_cmd.index('-c')+1]=str(OUT/'iq3-cache.cpp');compile_cmd[compile_cmd.index('-o')+1]=str(OUT/'iq3-cache.cpp.o');compile_cmd.insert(1,'-I'+str(OUT))
subprocess.run(compile_cmd,check=True,timeout=180)
link=next(cmd for cmd in prior['commands'] if isinstance(cmd,dict))
argv=[str(OUT/'iq3-cache.cpp.o') if x==str(BASE/'iq3-cache.cpp.o') else x for x in link['argv']]
argv[argv.index('-o')+1]=str(RUNTIME/'libggml-cpu.so.0.28.0');argv=[('-Wl,--dependency-file='+str(OUT/'link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in argv]
subprocess.run(argv,cwd=link['cwd'],check=True,timeout=120)
for alias in ['libggml-cpu.so','libggml-cpu.so.0']:
    if not (RUNTIME/alias).exists():(RUNTIME/alias).symlink_to('libggml-cpu.so.0.28.0')
record={'compile':compile_cmd,'link':{'argv':argv,'cwd':link['cwd']},'library':{'file':str(RUNTIME/'libggml-cpu.so.0.28.0'),'sha256':hashlib.sha256((RUNTIME/'libggml-cpu.so.0.28.0').read_bytes()).hexdigest()},'sources':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['iq3-cache.cpp','iq3-cache.h','compact-kernel.inc','iq-small-kernels.inc','placement.inc','launch_guard.py']}}
(OUT/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
print('Built isolated row-local cache overlay; no runtime profile changed.')
