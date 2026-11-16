#!/usr/bin/env python3
"""Compile a synthetic arithmetic probe; no model, API, GPU or shipped ELF runs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='new directory')
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    handoffs = repo/'archive/review-handoffs-not-for-readers'
    h = handoffs/'2026-11-16-current'
    e = h/'snapshot/llm-lab/exploration-20261112'
    incoming = handoffs/'2026-11-16-next-optimization/incoming/xeon_moe_review_2026-11-16/cpu'
    g = h/'snapshot/llm-lab/mxfp4-hybrid-20261102/hybrid/engine/ggml'
    local = Path(__file__).resolve().parent
    source = local/'iq3s_offset4_probe.cpp'
    dependencies = [source, local/'iq3s_offset4.inc', incoming/'iq_abs_rhs_probe.cpp',
                    e/'row-local/compact-kernel.inc', e/'row-local/iq-small-kernels.inc',
                    e/'iq-rhs/iq-rhs.cpp', e/'iq-rhs/iq-rhs.h', e/'iq-rhs/iq-math.inc']
    if sha(incoming/'iq_abs_rhs_probe.cpp') != '8db9910ab7f9ef2d4473535f5ce82b73de98a4377f1d350173b52a8fd2684cac':
        raise ValueError('Unexpected supplied generator/baseline source')
    includes = ['-I'+str(p) for p in [local,incoming,e/'row-local',e/'iq-rhs',g/'include',g/'src',g/'src/ggml-cpu']]
    flags = ['-O3','-std=gnu++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma',
             '-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections']
    records = []
    for command in [
        ['c++',*flags,*includes,str(source),str(e/'iq-rhs/iq-rhs.cpp'),'-Wl,--gc-sections','-lm','-o',str(out/'probe')],
        [str(out/'probe')],
        ['objdump','-d','-C',str(out/'probe')],
    ]:
        result = subprocess.run(command,capture_output=True,text=True,timeout=90)
        item = {'argv':command,'exit_code':result.returncode,'stderr':result.stderr}
        if command[0] == 'objdump':
            (out/'probe.asm').write_text(result.stdout)
            functions = re.split(r'(?m)(?=^[0-9a-f]+ <)',result.stdout)
            picked = [f for f in functions if re.match(r'^[0-9a-f]+ <offset4_fast_dot\(',f)]
            text = '\n'.join(picked)
            (out/'fast_dot.asm').write_text(text)
            item['selected_functions'] = len(picked)
            item['fast_path_instruction_counts'] = {name:len(re.findall(r'\b'+name+r'\b',text)) for name in
                                                   ['vpshufb','vpsignb','vpabsb','vpmaddubsw','vpaddw','vpsubw']}
        else:
            item['stdout'] = result.stdout
        records.append(item)
        if result.returncode:
            break
    passed = len(records)==3 and all(row['exit_code']==0 for row in records)
    report = {'scope':'Synthetic IQ3_S offset-nibble arithmetic and compiler inspection only; no performance measurement or backend integration.',
              'platform':platform.platform(), 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              'source_sha256':{str(p):sha(p) for p in dependencies}, 'commands':records,'passed':passed}
    if passed:
        report['observations']=json.loads(records[1]['stdout'])
    (out/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':passed,'result':str(out/'RESULT.json'),'observations':report.get('observations'),
                      'last_stderr':records[-1]['stderr'][:1500]},ensure_ascii=False))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
