#!/usr/bin/env python3
"""Compile the supplied abs helper inside a small new correctness-only probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True,help='new directory')
    args=ap.parse_args()
    repo=args.repository.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    handoffs=repo/'archive/review-handoffs-not-for-readers'
    n=handoffs/'2026-11-16-abs-operator'
    h=handoffs/'2026-11-16-current/snapshot/llm-lab'
    g=h/'mxfp4-hybrid-20261102/hybrid/engine/ggml'
    row=h/'exploration-20261112/row-local'
    source=Path(__file__).resolve().parent/'source_contract_probe.cpp'
    pinned={'iq-abs.cpp':'2055a9a91b3f26476e06750c6c9e5651999f3216d554cf4eef370e9da9b82eb5',
            'iq-abs.h':'d0f6beb5306f57207345b3312e02677ccb9735d35ebce7cac6dedab1c64825d6',
            'ggml-cpu.c':'b87089543cd8da780e7d0a0563c5c4056a4ea40a30dc5a1f17789abeb844e9db'}
    for name,expected in pinned.items():
        if sha(n/name)!=expected:raise ValueError('Unexpected source '+name)
    includes=['-I'+str(p) for p in [n,row,g/'include',g/'src',g/'src/ggml-cpu']]
    command=['c++','-std=gnu++17','-O1','-g','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma',
             '-fno-fast-math','-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all',
             '-fno-omit-frame-pointer','-ffunction-sections','-fdata-sections',*includes,str(source),
             '-Wl,--gc-sections','-o',str(out/'source_contract_probe')]
    records=[]
    probe_env=dict(os.environ)
    # LeakSanitizer cannot inspect processes in this ptraced execution environment.
    # Address/undefined sanitizers remain enabled; no leak-detection claim is made.
    probe_env['ASAN_OPTIONS']='detect_leaks=0'
    for argv in [command,[str(out/'source_contract_probe')]]:
        p=subprocess.run(argv,text=True,capture_output=True,timeout=60,env=probe_env)
        records.append({'argv':argv,'ASAN_OPTIONS':'detect_leaks=0','exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
        if p.returncode:break
    passed=len(records)==2 and all(p['exit_code']==0 for p in records)
    deps=[source,Path(__file__),*(n/name for name in pinned),row/'iq-small-kernels.inc',g/'include/ggml.h',g/'src/ggml-common.h']
    result={'scope':'Newly compiled helper-only functional audit; no engine/model/GPU/shipped ELF/performance run.',
            'platform':platform.platform(),'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'source_sha256':{str(p.resolve()):sha(p) for p in deps},'commands':records,'passed':passed}
    if passed:result['observations']=json.loads(records[-1]['stdout'])
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':passed,'result':str(out/'RESULT.json'),'observations':result.get('observations'),
                      'stderr':records[-1]['stderr'][:2000]},ensure_ascii=False))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
