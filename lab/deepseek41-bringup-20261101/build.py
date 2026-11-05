#!/usr/bin/env python3
"""User-run isolated upstream build for Ivy Bridge + Volta. No model startup."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

R=Path(__file__).resolve().parent
LAB=R.parent
REVISION='dde879bc4f954b809892fd0d660515f42536394a'
SOURCE=R/'source'
TOOLKIT=LAB/'v100-prep-20261024/cuda-12.9.2'

def run(cmd,label):
    print('\n'+label,flush=True)
    with (R/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in p.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            if p.wait():raise RuntimeError('Failed: '+label+' (see '+str(R/(label+'.log'))+')')
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()

def plan(jobs):
    options={
        'CMAKE_BUILD_TYPE':'Release','CMAKE_EXPORT_COMPILE_COMMANDS':'ON',
        'CMAKE_C_FLAGS':'-march=ivybridge -mtune=ivybridge',
        'CMAKE_CXX_FLAGS':'-march=ivybridge -mtune=ivybridge',
        'GGML_NATIVE':'OFF','GGML_AVX':'ON','GGML_AVX2':'OFF','GGML_AVX_VNNI':'OFF',
        'GGML_AVX512':'OFF','GGML_AVX512_VBMI':'OFF','GGML_AVX512_VNNI':'OFF','GGML_AVX512_BF16':'OFF',
        'GGML_FMA':'OFF','GGML_F16C':'ON','GGML_SSE42':'ON','GGML_BMI2':'OFF',
        'GGML_OPENMP':'ON','GGML_BLAS':'OFF','GGML_CPU_ALL_VARIANTS':'OFF','GGML_CPU_REPACK':'OFF',
        'GGML_CUDA':'ON','CUDAToolkit_ROOT':str(TOOLKIT),'CMAKE_CUDA_COMPILER':str(TOOLKIT/'bin/nvcc'),
        'CMAKE_CUDA_HOST_COMPILER':'/usr/bin/g++','CMAKE_CUDA_ARCHITECTURES':'70-real',
        'CMAKE_CUDA_FLAGS':'-Xcompiler=-march=ivybridge','CMAKE_BUILD_RPATH':str(TOOLKIT/'lib'),
        'GGML_CUDA_GRAPHS':'ON','GGML_CUDA_FA':'ON','GGML_CUDA_FA_QUANTS':'f16-f16',
        'GGML_CUDA_NCCL':'OFF','GGML_CUDA_NO_VMM':'ON','GGML_CUDA_FORCE_MMQ':'OFF',
        'GGML_CUDA_FORCE_CUBLAS':'OFF','GGML_RPC':'OFF',
        'LLAMA_BUILD_TESTS':'OFF','LLAMA_BUILD_SERVER':'ON','LLAMA_BUILD_UI':'OFF',
        'LLAMA_BUILD_MTMD':'OFF','LLAMA_OPENSSL':'OFF'}
    return [['cmake','-S',str(SOURCE),'-B',str(R/'build')]+[f'-D{k}={v}' for k,v in options.items()],
            ['cmake','--build',str(R/'build'),'-j',str(jobs),'--target','llama-server']]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--print-only',action='store_true')
    args=ap.parse_args();assert 1<=args.jobs<=12
    commands=plan(args.jobs)
    (R/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    if args.print_only:
        print(json.dumps({'source':str(SOURCE),'revision':REVISION,'commands':commands},indent=2));return
    for service in ['llm-openai-api.service','llm-deepseek41-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',service]).returncode==0:
            raise RuntimeError(service+' is active; stop it yourself before compiling.')
    if not (TOOLKIT/'bin/nvcc').exists():raise RuntimeError('Prepared CUDA12.9 toolkit missing')
    version=subprocess.check_output([str(TOOLKIT/'bin/nvcc'),'--version'],text=True)
    if 'release 12.9,' not in version:raise RuntimeError('Expected CUDA12.9 for Volta')
    with (LAB/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not SOURCE.exists():
            run(['git','init',str(SOURCE)],'git-init')
            run(['git','-C',str(SOURCE),'remote','add','origin','https://github.com/ggml-org/llama.cpp.git'],'git-remote')
        if not (SOURCE/'.git').exists():raise RuntimeError('Existing source directory is not a git checkout; refusing overwrite')
        status=subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
        if status.strip():raise RuntimeError('Source has local changes; refusing to overwrite')
        current=subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],capture_output=True,text=True)
        if current.returncode or current.stdout.strip()!=REVISION:
            run(['git','-C',str(SOURCE),'fetch','--depth','1','origin',REVISION],'git-fetch')
            run(['git','-C',str(SOURCE),'checkout','--detach',REVISION],'git-checkout')
        assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==REVISION
        run(commands[0],'configure');run(commands[1],'build')
        binary=R/'build/bin/llama-server';env=dict(os.environ,LD_LIBRARY_PATH=str(R/'build/bin')+':'+str(TOOLKIT/'lib'))
        env.pop('LD_PRELOAD',None)
        help_text=subprocess.check_output([str(binary),'--help'],env=env,text=True,stderr=subprocess.STDOUT)
        (R/'server-help.txt').write_text(help_text)
        for flag in ['--cpu-moe','--cache-prompt','--cache-ram','--ctx-checkpoints','--no-repack','--load-mode','--lazy-mode']:
            if flag not in help_text:raise RuntimeError('Missing required option: '+flag)
        version=subprocess.check_output([str(binary),'--version'],env=env,text=True,stderr=subprocess.STDOUT)
        record={'built':True,'revision':REVISION,'version':version,'unix':time.time(),
                'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                'real_model_tested':False,'commands':commands}
        (R/'BUILD_READY.json').write_text(json.dumps(record,indent=2)+'\n')
    print('\nビルド完了。モデルは起動していません。次: sudo python3 '+str(R/'start.py'),flush=True)

if __name__=='__main__':main()
