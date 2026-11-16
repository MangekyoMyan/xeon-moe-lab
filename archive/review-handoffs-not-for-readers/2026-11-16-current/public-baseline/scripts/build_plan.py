#!/usr/bin/env python3
"""Print the fixed Ivy Bridge/Volta reconstruction plan. Execution is explicit and user-run."""
from pathlib import Path
import argparse,json,os,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--cuda-root',type=Path,required=True);p.add_argument('--jobs',type=int,default=4);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not 1<=a.jobs<=12:raise SystemExit('jobs must be1..12')
    source=ROOT/'work/accepted-source';build=ROOT/'work/build';cuda=a.cuda_root.resolve()
    options={'CMAKE_BUILD_TYPE':'Release','CMAKE_EXPORT_COMPILE_COMMANDS':'ON','CMAKE_C_FLAGS':'-march=ivybridge -mtune=ivybridge -mno-avx2 -mno-fma','CMAKE_CXX_FLAGS':'-march=ivybridge -mtune=ivybridge -mno-avx2 -mno-fma',
     'GGML_NATIVE':'OFF','GGML_AVX':'ON','GGML_F16C':'ON','GGML_SSE42':'ON','GGML_AVX2':'OFF','GGML_FMA':'OFF','GGML_BMI2':'OFF','GGML_AVX_VNNI':'OFF','GGML_AVX512':'OFF','GGML_AVX512_VBMI':'OFF','GGML_AVX512_VNNI':'OFF','GGML_AVX512_BF16':'OFF',
     'GGML_CPU_REPACK':'OFF','GGML_CPU_ALL_VARIANTS':'OFF','GGML_OPENMP':'ON','GGML_BLAS':'OFF','GGML_CUDA':'ON','CUDAToolkit_ROOT':str(cuda),'CMAKE_CUDA_COMPILER':str(cuda/'bin/nvcc'),'CMAKE_CUDA_HOST_COMPILER':'/usr/bin/g++',
     'CMAKE_CUDA_ARCHITECTURES':'70-real','CMAKE_CUDA_FLAGS':'-Xcompiler=-march=ivybridge','CMAKE_BUILD_RPATH':str(cuda/'lib')+';'+str(cuda/'lib64'),'GGML_CUDA_GRAPHS':'ON','GGML_CUDA_FA':'ON','GGML_CUDA_FA_QUANTS':'f16-f16','GGML_CUDA_NCCL':'OFF','GGML_CUDA_NO_VMM':'ON',
     'GGML_CUDA_FORCE_MMQ':'OFF','GGML_CUDA_FORCE_CUBLAS':'OFF','GGML_RPC':'OFF','LLAMA_BUILD_TESTS':'OFF','LLAMA_BUILD_SERVER':'ON','LLAMA_BUILD_UI':'OFF','LLAMA_BUILD_MTMD':'OFF','LLAMA_OPENSSL':'OFF'}
    commands=[['cmake','-S',str(source),'-B',str(build)]+[f'-D{k}={v}' for k,v in options.items()],['cmake','--build',str(build),'-j',str(a.jobs),'--target','llama-server']]
    print(json.dumps(commands,indent=2))
    if not a.execute:return
    subprocess.run(['python3',str(ROOT/'scripts/prepare_source.py'),'--verify'],check=True)
    for unit in ['llm-deepseek41-api.service','llm-openai-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',unit]).returncode==0:raise SystemExit('API active; this archival helper does not stop it: '+unit)
    if not (cuda/'bin/nvcc').is_file():raise SystemExit('CUDA toolkit missing')
    if 'release 12.9,' not in subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True):raise SystemExit('This frozen profile expects CUDA12.9')
    for cmd in commands:subprocess.run(cmd,check=True)
    print('Build finished in work/build. No model startup or production deployment.')
if __name__=='__main__':main()
