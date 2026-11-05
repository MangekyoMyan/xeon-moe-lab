#!/usr/bin/python3
"""Check each existing option against the saved independent CPU oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
R29=ROOT.parent/'moe-engine-20261029'
parser=argparse.ArgumentParser();parser.add_argument('--simple',action='store_true');args=parser.parse_args()
RT=ROOT/'variants/simple-prefill/runtime' if args.simple else R29/'variants/hybrid-reuse-v1/runtime'
OLD=R29/'experiments/hybrid-audit-agent/correctness'
row=next(x for x in json.loads((OLD/'results.json').read_text()) if x['suite']=='existing' and x['mode']=='cpu')
oracle=(OLD/'existing-cpu.bin').read_bytes()
assert hashlib.sha256(oracle).hexdigest()==row['sha256']
settings={'pair':{'GGML_XV_PREFILL_PAIR':'1'},'rows4':{'GGML_CPU_Q8_MOE_ROWS4':'1'},
          'token-axis':{'GGML_XV_PREFILL_TOKEN_AXIS':'1'}}
if args.simple:
    settings={'default':{},'phase':{'GGML_XV_ADAPTIVE_PREFILL_INTERVAL':'1'},'swap8':{'GGML_XV_PREFILL_SWAP_GAIN':'8'},'swap16':{'GGML_XV_PREFILL_SWAP_GAIN':'16'}}
basefolder=ROOT/('operator-checks-simple' if args.simple else 'operator-checks')
results=[]
for name,extra in settings.items():
    folder=basefolder/name
    folder.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
    env.update(LD_LIBRARY_PATH=str(RT)+':'+str(ROOT.parent/'v100-prep-20261024/cuda-12.9.2/lib'),
        GGML_XV_PLUGIN=str(RT/'libxv-experiment.so'),GGML_XV_EXACT_Q8='1',GGML_XV_SHARE_DECODE_GRAPHS='1',
        GGML_XV_WHOLE_MLP='1',GGML_XV_GATE_UP_ONLY='1',GGML_XV_MLP_CPU_PAIR='1',GGML_CPU_MOE_PAIR='0',
        GGML_XV_TEST_MUTABLE='1',GGML_XV_TEST_SLOTS='3',GGML_XV_WORKER_CPU='18',GGML_XV_COPY_CPUS='0,1,6,7',
        GGML_XV_PINNED_ACTIVATIONS='1',GGML_XV_CHECK_PINNED='1')
    env.update(extra)
    with (folder/'stdout.txt').open('w') as out,(folder/'stderr.txt').open('w') as err:
        p=subprocess.run(['numactl','--interleave=0,1',str(R29/'tests/prefill-strided-cases'),str(folder/'output.bin')],env=env,stdout=out,stderr=err)
    raw=(folder/'output.bin').read_bytes() if (folder/'output.bin').exists() else b''
    result=dict(option=name,returncode=p.returncode,CPU_bitexact=p.returncode==0 and raw==oracle,
                cases=184 if raw==oracle else None,sha256=hashlib.sha256(raw).hexdigest(),environment_changes=extra)
    lines=(folder/'stderr.txt').read_text().splitlines()
    swaps=[json.loads(x.split(' ',1)[1]) for x in lines if x.startswith('XV_PREFILL_SWAP ')]
    result['prefill_swaps']=sum(x['operations'] for x in swaps)
    result['extra_gpu_positions']=sum(x['extra_gpu_positions'] for x in swaps)
    results.append(result)
    print(json.dumps(result),flush=True)
(basefolder/'results.json').write_text(json.dumps(results,indent=2)+'\n')
