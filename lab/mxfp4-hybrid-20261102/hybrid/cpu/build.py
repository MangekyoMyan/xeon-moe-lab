#!/usr/bin/env python3
"""Build only requested CPU targets; no sudo, CUDA, downloads or model startup."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['portable','ivybridge','native'],default='portable')
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--configure-only',action='store_true')
    ap.add_argument('--target',choices=['probe','engine','arch-tests','backend-tests'],default='probe')
    args=ap.parse_args()
    if not 1<=args.jobs<=32:ap.error('--jobs must be1..32')
    if not shutil.which('cmake'):raise SystemExit('Install CMake and a C/C++17 compiler in your Linux environment first.')
    folder=ROOT/('build-cpu-'+args.profile)
    commands=[['cmake','-S',str(HERE),'-B',str(folder),'-DCMAKE_BUILD_TYPE=Release','-DREVIEW_PROFILE='+args.profile]]
    targets={'probe':['cpu-quant-probe'],'engine':['llama-server','llama-cli'],
             'arch-tests':['test-llama-archs','test-llama-context'],'backend-tests':['test-backend-ops','test-quantize-perf']}
    if not args.configure_only:commands.append(['cmake','--build',str(folder),'--parallel',str(args.jobs),'--target',*targets[args.target]])
    folder.mkdir(exist_ok=True)
    (folder/'review-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(ROOT))
    for command in commands:subprocess.run(command,env=env,check=True)
    print('Output:',folder/'bin')


if __name__=='__main__':main()
