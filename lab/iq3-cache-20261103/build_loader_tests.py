from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent;L=R.parent;E=L/'deepseek41-bringup-20261101';S=E/'source'
for name in ['make_fixture','test-loader']:
    cmd=['g++','-O2','-std=c++17','-I'+str(R),'-I'+str(S/'include'),'-I'+str(S/'ggml/include'),'-I'+str(S/'ggml/src'),str(R/(name+'.cpp')),
     '-L'+str(R/'runtime'),'-L'+str(E/'build/bin'),'-Wl,-rpath,'+str(R/'runtime'),'-Wl,-rpath,'+str(L/'gpu-stream-20261103/runtime'),'-Wl,-rpath,'+str(E/'build/bin'),
     '-lllama','-lggml-cpu','-lggml-base','-lggml-cuda','-lggml','-o',str(R/name)]
    subprocess.run(cmd,check=True,timeout=60)
subprocess.run([str(R/'make_fixture'),str(R/'tiny-iq3.gguf')],check=True)
