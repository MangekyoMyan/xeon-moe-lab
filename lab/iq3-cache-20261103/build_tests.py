from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101'
S=L/'mxfp4-hybrid-20261102/hybrid/engine/ggml';cmd=['g++','-O3','-std=c++17','-march=ivybridge','-mno-avx2','-mno-fma','-ffp-contract=off','-fopenmp',
'-I'+str(R),'-I'+str(S/'include'),'-I'+str(S/'src'),str(R/'test-cache.cpp'),'-L'+str(R/'runtime'),'-L'+str(E/'build/bin'),
'-Wl,-rpath,'+str(R/'runtime'),'-Wl,-rpath,'+str(L/'gpu-stream-20261103/runtime'),'-Wl,-rpath,'+str(E/'build/bin'),'-lggml-cpu','-lggml-base','-o',str(R/'test-cache')]
subprocess.run(cmd,check=True,timeout=90);(R/'TEST_BUILD.json').write_text(json.dumps(cmd,indent=2)+'\n')
