from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';C=L/'v100-prep-20261024/cuda-12.9.2'
commands=[
 [str(C/'bin/nvcc'),'-O2','-std=c++17','-arch=sm_70','-Xcompiler','-fPIC','-c',str(R/'kernels.cu'),'-o',str(R/'kernels.o')],
 ['g++','-O2','-std=c++17','-pthread','-I'+str(E/'source/ggml/include'),'-I'+str(C/'include'),str(R/'split_probe.cpp'),str(R/'kernels.o'),
 '-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(E/'build/bin'),'-L'+str(C/'lib'),'-Wl,-rpath,'+str(C/'lib'),'-lggml-cpu','-lggml-cuda','-lggml-base','-lggml','-lcudart','-o',str(R/'split_probe')]]
for cmd in commands:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=90);print(p.returncode,p.stderr[-5000:],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
(R/'BUILD.json').write_text(json.dumps({'commands':commands,'sha256':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['split_probe.cpp','kernels.cu','split_probe']}},indent=2)+'\n')
