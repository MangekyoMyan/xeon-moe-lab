#!/usr/bin/env python3
"""Compile a CPU-only synthetic contract probe from the shared snapshot.

No model, server, packaged binary, API, GPU, or controller is executed.
The full reference C file is compiled with function sections; the linker retains
only the MXFP4 dot used by the probe. The source checkout is not modified.
"""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.snapshot.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    exp=root/'snapshot/llm-lab/exploration-20261112'
    engine=root/'snapshot/llm-lab/mxfp4-hybrid-20261102/hybrid/engine'
    ggml=engine/'ggml'
    mx=exp/'mx-prepack/release-candidate-a/sources'
    source=ggml/'src/ggml-cpu/arch/x86/quants.c'
    probe=Path(__file__).with_name('mx_contract_probe.cpp').resolve()
    common=['-O3','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections']
    include=['-I'+str(x) for x in [ggml/'include',ggml/'src',ggml/'src/ggml-cpu',mx]]
    commands=[
        ['cc',*common,'-std=gnu11',*include,'-c',str(source),'-o',str(out/'reference.o')],
        ['c++',*common,'-std=gnu++17',*include,'-c',str(mx/'mx-rhs.cpp'),'-o',str(out/'mx.o')],
        ['c++',*common,'-std=gnu++17',*include,str(probe),str(out/'reference.o'),str(out/'mx.o'),'-Wl,--gc-sections','-lm','-o',str(out/'probe')],
        [str(out/'probe')],
    ]
    records=[]
    for cmd in commands:
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
        records.append({'argv':cmd,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:break
    result={'platform':platform.platform(),'source_sha256':{str(p.relative_to(root)) if p.is_relative_to(root) else str(p):digest(p) for p in [source,mx/'mx-rhs.cpp',mx/'mx-rhs.h',mx/'mx-math.inc',probe]},'commands':records,'passed':len(records)==4 and records[-1]['exit_code']==0}
    if result['passed']:result['observations']=json.loads(records[-1]['stdout'])
    (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out/'RESULT.json'),'passed':result['passed'],'last_stderr':records[-1]['stderr'][:2500],'observations':result.get('observations')},ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
