#!/usr/bin/env python3
"""Untimed route/stride and dispatch checks. Never run alongside timing suites."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

R=Path(__file__).resolve().parent
BUNDLE=R.parent/'pro-review-received-20261102/deepseek-cpu-review-result'
OUT=R/'diagnostics';OUT.mkdir(exist_ok=False)
BIN=R/'mxfp4/build-cpu-ivybridge/bin'
commands=[]
def run(cmd,**kwargs):
    commands.append(cmd)
    return subprocess.run(cmd,check=True,capture_output=True,text=True,**kwargs)

route=OUT/'route-test'
run(['g++','-std=c++17','-O3','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma',
     '-I'+str(R/'mxfp4/engine/ggml/include'),str(BUNDLE/'route_test/test.cpp'),'-L'+str(BIN),
     '-Wl,-rpath,'+str(BIN),'-lggml-cpu','-lggml-base','-o',str(route)])
result=run([str(route)],timeout=60)
(OUT/'route.jsonl').write_text(result.stdout);(OUT/'route.stderr').write_text(result.stderr)
rows=[json.loads(x) for x in result.stdout.splitlines()]
assert rows[-1]['cases']==64 and rows[-1]['failures']==0
route_summary={'cases':64,'outputs':sum(x['outputs'] for x in rows[:-1]),
               'bit_mismatches':sum(x['bit_mismatches'] for x in rows[:-1]),'pass':True}

shim=OUT/'count-pair-calls.so'
run(['gcc','-std=c11','-O2','-shared','-fPIC',str(R/'count_pair_calls.c'),'-ldl','-o',str(shim)])
base=['numactl','--interleave=0,1','taskset','-c','0','env','LD_PRELOAD='+str(shim),str(BIN/'cpu-quant-probe'),
      '--type','mxfp4','--threads','1','--strict-cpu','--mode','up','--k','512','--m','33',
      '--experts','2','--topk','2','--routing','shared','--warmups','0','--repeats','1']
env=dict(os.environ);env.pop('LD_PRELOAD',None);dispatch=[]
for tokens,reference,expected in [(1,False,0),(2,False,66),(3,False,66),(2,True,0)]:
    label=f'tokens{tokens}-ref{int(reference)}'
    result=run(base+['--tokens',str(tokens)]+(['--use-ref'] if reference else []),env=env,timeout=30)
    (OUT/(label+'.stdout')).write_text(result.stdout);(OUT/(label+'.stderr')).write_text(result.stderr)
    items=[json.loads(line) for line in result.stderr.splitlines() if 'paired_kernel_calls' in line]
    assert len(items)==1 and items[0]['paired_kernel_calls']==expected,(label,result.stderr)
    dispatch.append({'tokens':tokens,'use_ref':reference,'expected_calls':expected,'observed_calls':items[0]['paired_kernel_calls']})
record={'route':route_summary,'dispatch':dispatch,'performance_timing':False,
        'CPU_library_sha256':hashlib.sha256((BIN/'libggml-cpu.so.0').read_bytes()).hexdigest(),
        'commands':commands}
(OUT/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'route':route_summary,'dispatch':dispatch,'passed':True}),flush=True)
