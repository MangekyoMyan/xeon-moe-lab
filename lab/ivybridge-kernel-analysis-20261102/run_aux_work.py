#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pmu',R/'run_pmu.py')
pmu=importlib.util.module_from_spec(spec);spec.loader.exec_module(pmu)
ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['positive','micro'],required=True);a=ap.parse_args()
if a.part=='positive':
    out=R/'positive-controls';out.mkdir(exist_ok=False)
    binary=out/'transition-control'
    build=['gcc','-O2','-mavx','-mno-avx2','-mno-fma',str(R/'transition_control.c'),'-o',str(binary)]
    subprocess.run(build,check=True,timeout=30)
    asm=subprocess.check_output(['objdump','-d','-M','intel',str(binary)],text=True)
    (out/'control.asm').write_text(asm)
    records=[]
    for mode in ['vex','legacy']:
        cmd=['perf','stat','-x',',','-e','{cycles:u,instructions:u,other_assists.avx_to_sse:u,other_assists.sse_to_avx:u}',
             '-e','task-clock','-o',str(out/(mode+'.csv')),'--','taskset','-c','0',str(binary),mode]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=15)
        events=pmu.parse_perf(out/(mode+'.csv'),'assists')
        record={'mode':mode,'iterations':200000,'command':cmd,'events':events,'stdout':result.stdout}
        forward=events['other_assists.avx_to_sse:u']['value'];reverse=events['other_assists.sse_to_avx:u']['value']
        if mode=='vex':assert forward<10 and reverse<10
        else:assert forward>180000 and reverse>180000
        records.append(record)
        print(json.dumps({'mode':mode,'AVX_to_SSE':forward,'SSE_to_AVX':reverse,'control_passed':True}),flush=True)
    (out/'RESULT.json').write_text(json.dumps({'build':build,'records':records,'passed':True},indent=2)+'\n')
else:
    for variant in ['pairlanes','cols4']:
        cmd=['python3',str(R/'micro/run_review.py'),'--profile','ivybridge','--optimization','O3',
             '--variant',variant,'--mode','all','--cpu','0','--trials','9','--suite','full']
        subprocess.run(cmd,check=True,timeout=100)
