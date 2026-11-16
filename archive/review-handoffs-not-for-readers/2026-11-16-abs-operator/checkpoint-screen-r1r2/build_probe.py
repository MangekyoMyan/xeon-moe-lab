from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parents[1]
g=LAB/'mxfp4-hybrid-20261102/hybrid/engine/ggml'
cmd=['g++','-O3','-std=c++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-fopenmp','-Wall','-Wextra','-I'+str(g/'include'),'-I'+str(g/'src'),'-I'+str(LAB/'exploration-20261112/row-local'),str(R/'probe.cpp'),'-L'+str(R/'runtime'),'-L'+str(LAB/'gpu-stream-20261103/runtime'),'-L'+str(LAB/'deepseek41-bringup-20261101/build/bin'),'-Wl,-rpath,'+str(R/'runtime')+':'+str(LAB/'gpu-stream-20261103/runtime')+':'+str(LAB/'deepseek41-bringup-20261101/build/bin'),'-lggml-cpu','-lggml-base','-lggml','-ldl','-o',str(R/'probe')]
subprocess.run(cmd,check=True,timeout=180)
(R/'PROBE_BUILD.json').write_text(json.dumps({'argv':cmd,'source_sha256':hashlib.sha256((R/'probe.cpp').read_bytes()).hexdigest(),'probe_sha256':hashlib.sha256((R/'probe').read_bytes()).hexdigest()},indent=2)+'\n')
print('Built actual ggml operator probe')
