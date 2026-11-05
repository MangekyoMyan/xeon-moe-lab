from pathlib import Path
import hashlib
import json
import subprocess
R=Path(__file__).resolve().parent;LAB=R.parent
S=LAB/'mxfp4-hybrid-20261102/hybrid/engine/ggml'
runtime=LAB/'gpu-stream-20261103/runtime';original=LAB/'deepseek41-bringup-20261101/build/bin'
common=['g++','-std=c++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-fopenmp',
 '-I'+str(S/'include'),'-I'+str(S/'src'),str(R/'predecode_probe.cpp'),'-L'+str(runtime),'-L'+str(original),
 '-Wl,-rpath,'+str(runtime),'-Wl,-rpath,'+str(original),'-lggml-cpu','-lggml-base']
records=[]
for name,flags in [('predecode_probe',['-O3']),('check-sanitized',['-O1','-g','-fsanitize=address,undefined'])]:
    command=[*common,*flags,'-o',str(R/name)];subprocess.run(command,check=True,timeout=60)
    records.append({'name':name,'command':command,'binary_sha256':hashlib.sha256((R/name).read_bytes()).hexdigest()})
(R/'BUILD.json').write_text(json.dumps({'source_sha256':hashlib.sha256((R/'predecode_probe.cpp').read_bytes()).hexdigest(),
 'native_CPU_library_sha256':hashlib.sha256((runtime/'libggml-cpu.so.0.28.0').read_bytes()).hexdigest(),'builds':records},indent=2)+'\n')
