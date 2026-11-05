#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

R=Path(__file__).resolve().parent;M=R/'micro'
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['original4','looped4','sequential4'],required=True)
ap.add_argument('--portable',action='store_true');a=ap.parse_args()
out=M/'build'/(a.variant+('-portable' if a.portable else ''));out.mkdir(parents=True,exist_ok=False)
flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fno-lto']
flags+=['-march=x86-64','-mno-avx','-mno-avx2','-mno-fma','-mno-f16c'] if a.portable else ['-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma']
commands=[['g++',*flags,'-Dcandidate_batch=review_pair_batch','-Dcandidate_strided=review_pair_strided',
 '-Dcandidate_dot_cols=review_pair_dot_cols','-c',str(M/'candidate-pairlanes.cpp'),'-o',str(out/'pair.o')],
 ['g++',*flags,'-DREVIEW_COLS=4','-DHYBRID_KERNEL="'+str(R/'kernels'/(a.variant+'.inc'))+'"',
 str(M/'compare.cpp'),str(M/'hybrid.cpp'),str(out/'pair.o'),'-o',str(out/'compare')]]
for cmd in commands:subprocess.run(cmd,check=True,timeout=45)
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
env=dict(os.environ,REVIEW_CPU='0')
with (out/'results.jsonl').open('w') as log:
    modes=[['check']]
    if not a.portable:
        modes += [['bench',str(k),str(m),str(c),'7'] for k,m in [(5120,2304),(2304,5120)] for c in [1,2,3,4,5,6,7,8,16,17]]
    for args in modes:
        p=subprocess.run([str(out/'compare'),*args],env=env,check=True,capture_output=True,text=True,timeout=30)
        for line in p.stdout.splitlines():
            item=json.loads(line);item['variant']=a.variant;item['portable']=a.portable
            log.write(json.dumps(item)+'\n');log.flush()
            if item['kind']=='correctness':print(json.dumps(item),flush=True)
    symbols=[line.split()[-1] for line in subprocess.check_output(['nm',str(out/'compare')],text=True).splitlines() if 'ggml_vec_dot_mxfp4_q8_0_cols4' in line]
    for i,sym in enumerate(symbols):
        asm=subprocess.check_output(['objdump','-d','-M','intel','--disassemble='+sym,str(out/'compare')],text=True)
        (out/('kernel-'+str(i)+'.asm')).write_text(asm)
print(json.dumps({'variant':a.variant,'completed':True,'binary_sha256':hashlib.sha256((out/'compare').read_bytes()).hexdigest(),'results':str(out/'results.jsonl')}),flush=True)
