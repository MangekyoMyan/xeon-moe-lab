from pathlib import Path
import subprocess,json,hashlib
r=Path(__file__).resolve().parent;lab=r.parent.parent
ggml=lab/'mxfp4-hybrid-20261102/hybrid/engine/ggml';bin=lab/'deepseek41-bringup-20261101/build/bin'
cmd=['g++','-O3','-std=c++17','-march=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math',
     '-ffp-contract=off','-fopenmp','-I'+str(ggml/'include'),'-I'+str(ggml/'src'),str(r/'test-bias.cpp'),
     '-L'+str(bin),'-Wl,-rpath,'+str(bin),'-lggml-cpu','-lggml-base','-o',str(r/'test-bias')]
subprocess.run(cmd,check=True,timeout=120)
(r/'BUILD.json').write_text(json.dumps({'command':cmd,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.inc')},'binary_sha256':hashlib.sha256((r/'test-bias').read_bytes()).hexdigest()},indent=2)+'\n')
subprocess.run([str(r/'test-bias')],check=True,timeout=20)
