from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101';CUDA=LAB/'v100-prep-20261024/cuda-12.9.2';OUT=R/'cuda-sort'
cmd=['g++','-O2','-std=c++17','-I'+str(ENGINE/'source/ggml/include'),'-I'+str(CUDA/'include'),str(OUT/'test-gpu.cpp'),'-L'+str(ENGINE/'build/bin'),'-Wl,-rpath,'+str(ENGINE/'build/bin'),'-L'+str(CUDA/'lib'),'-Wl,-rpath,'+str(CUDA/'lib'),'-lggml-cuda','-lggml-cpu','-lggml-base','-lggml','-lcudart','-ldl','-lcrypto','-o',str(OUT/'test-gpu')]
subprocess.run(cmd,check=True,timeout=180);(OUT/'GPU_TEST_BUILD.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((OUT/'test-gpu.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((OUT/'test-gpu').read_bytes()).hexdigest()},indent=2)+'\n');print('Built isolated CUDA sort operator fixture')
