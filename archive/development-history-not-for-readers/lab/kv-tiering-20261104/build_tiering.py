from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';s=E/'source';rt=L/'iq-cache-next-20261103/runtime'
cmd=['g++','-O3','-std=c++17','-march=ivybridge','-mno-avx2','-mno-fma','-fopenmp','-I'+str(s/'ggml/include'),str(R/'tiering_probe.cpp'),'-L'+str(rt),'-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(rt),'-Wl,-rpath,'+str(L/'gpu-stream-20261103/runtime'),'-Wl,-rpath,'+str(E/'build/bin'),'-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-o',str(R/'tiering_probe')]
subprocess.run(cmd,check=True,timeout=60);(R/'TIERING_BUILD.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((R/'tiering_probe.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((R/'tiering_probe').read_bytes()).hexdigest()},indent=2)+'\n')
