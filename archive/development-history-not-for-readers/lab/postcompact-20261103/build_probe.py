from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent;L=R.parent;S=L/'mxfp4-hybrid-20261102/hybrid/engine/ggml';rt=L/'iq3-cache-20261103/runtime';base=L/'deepseek41-bringup-20261101/build/bin'
cmd=['g++','-std=c++17','-march=ivybridge','-mno-avx2','-mno-fma','-ffp-contract=off','-fno-fast-math','-fopenmp',
 '-I'+str(S/'include'),'-I'+str(S/'src'),str(R/'iq-small-probe.cpp'),'-L'+str(rt),'-L'+str(base),'-Wl,-rpath,'+str(rt),'-Wl,-rpath,'+str(base),'-lggml-cpu','-lggml-base']
records=[]
for name,flags in [('iq-small-probe',['-O3']),('iq-small-sanitized',['-O1','-g','-fsanitize=address,undefined'])]:
 c=cmd+flags+['-o',str(R/name)];subprocess.run(c,check=True,timeout=90);records.append({'name':name,'command':c,'sha256':hashlib.sha256((R/name).read_bytes()).hexdigest()})
(R/'PROBE_BUILD.json').write_text(json.dumps({'builds':records,'sources':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['iq-small-probe.cpp','iq-small-kernels.inc']}},indent=2)+'\n')
