#!/usr/bin/env python3
"""Compile a proposed integration contract fixture; no model/service/GPU is used."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    root=args.repository.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    h=root/'archive/review-handoffs-not-for-readers/2026-11-16-current'
    n=root/'archive/review-handoffs-not-for-readers/2026-11-16-next-optimization'
    e=h/'snapshot/llm-lab/exploration-20261112'
    g=h/'snapshot/llm-lab/mxfp4-hybrid-20261102/hybrid/engine/ggml'
    own=Path(__file__).with_name('abs_integration_contract.cpp').resolve()
    upstream=n/'incoming/xeon_moe_review_2026-11-16/cpu/iq_abs_rhs_probe.cpp'
    if hashlib.sha256(upstream.read_bytes()).hexdigest()!='8db9910ab7f9ef2d4473535f5ce82b73de98a4377f1d350173b52a8fd2684cac':
        raise RuntimeError('Expected exact incoming probe source')
    includes=['-I'+str(p) for p in [g/'include',g/'src',g/'src/ggml-cpu',e/'row-local',e/'iq-rhs',upstream.parent]]
    flags=['-O2','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections']
    sources=[g/'src/ggml-cpu/arch/x86/quants.c',g/'src/ggml-quants.c',e/'iq-rhs/iq-rhs.cpp',own,upstream,e/'row-local/compact-kernel.inc',e/'row-local/iq-small-kernels.inc']
    commands=[['cc',*flags,'-std=gnu11',*includes,'-c',str(sources[0]),'-o',str(out/'reference.o')],
              ['cc',*flags,'-std=gnu11',*includes,'-c',str(sources[1]),'-o',str(out/'quantize.o')],
              ['c++',*flags,'-std=gnu++17',*includes,str(own),str(sources[2]),str(out/'reference.o'),str(out/'quantize.o'),'-Wl,--gc-sections','-pthread','-lm','-o',str(out/'contract_probe')],
              [str(out/'contract_probe')]]
    records=[]
    for command in commands:
        cp=subprocess.run(command,text=True,capture_output=True,timeout=90)
        records.append({'argv':command,'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
        if cp.returncode:break
    passed=len(records)==len(commands) and records[-1]['returncode']==0
    result={'scope':'Proposed shared/private preparation protocol, real std::threads with CV publication, pinned kernels/quantizer. Not ggml threadpool/allocator/backend integration or performance measurement.',
            'platform':platform.platform(),'source_sha256':{str(p.relative_to(root)) if p.is_relative_to(root) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'commands':records,'passed':passed}
    if passed:result['observations']=json.loads(records[-1]['stdout'])
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':passed,'output':str(out/'RESULT.json'),'observations':result.get('observations'),'last_stderr':records[-1]['stderr'][:2500]},ensure_ascii=False))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
