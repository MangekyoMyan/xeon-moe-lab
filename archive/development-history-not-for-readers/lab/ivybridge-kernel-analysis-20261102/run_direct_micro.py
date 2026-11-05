#!/usr/bin/env python3
"""Compare pair and four-column implementations within the same binary/input."""
from pathlib import Path
import json
import os
import subprocess

R=Path(__file__).resolve().parent
M=R/'micro';out=M/'direct-pair-vs-four';out.mkdir(exist_ok=False)
flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-fno-lto',
       '-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma']
commands=[
 ['g++',*flags,'-Dcandidate_batch=review_pair_batch','-Dcandidate_strided=review_pair_strided',
  '-Dcandidate_dot_cols=review_pair_dot_cols','-c',str(M/'candidate-pairlanes.cpp'),'-o',str(out/'pair.o')],
 ['g++',*flags,'-DREVIEW_COLS=4','-c',str(M/'candidate.cpp'),'-o',str(out/'four.o')],
 ['g++',*flags,'-DREVIEW_COLS=4',str(M/'review_compare.cpp'),str(out/'pair.o'),str(out/'four.o'),'-o',str(out/'compare')]
]
for cmd in commands:subprocess.run(cmd,check=True,timeout=45)
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
env=dict(os.environ,REVIEW_CPU='0')
with (out/'results.jsonl').open('w') as f:
    for k,m in [(5120,2304),(2304,5120)]:
        for columns in [1,2,3,4,8,16]:
            cmd=[str(out/'compare'),'bench',str(k),str(m),str(columns),'9']
            result=subprocess.run(cmd,check=True,capture_output=True,text=True,env=env,timeout=30)
            row=json.loads(result.stdout);row['direct_pair_four_bit_exact']=True
            text=json.dumps(row);f.write(text+'\n');f.flush();print(text,flush=True)
