#!/usr/bin/env python3
"""Replay recorded inputs after relocation, without hardware benchmarks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from run_deep import ORDER

SRC=Path(__file__).resolve().parent
LAB=SRC.parent
OUT=LAB/'results/hybrid-sim-deep-20261023'
ORIGINAL='/home/codex/projects/llm-lab/'

def relocate(path):
    if path.startswith(ORIGINAL):return str(LAB/path[len(ORIGINAL):])
    if Path(path).is_file():return path
    raise ValueError('Cannot locate recorded input: '+path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    cases=[(OUT,'smoke')] if args.smoke else [(OUT,m) for m in ['smoke','screen','sweep','diagnostics','cpu_sensitivity']]+[(OUT/'synthetic','synthetic')]
    report=[]
    with tempfile.TemporaryDirectory(prefix='deep-sim-replay-') as temporary:
        tmp=Path(temporary);binary=tmp/'sim';details=tmp/'details';details.mkdir()
        subprocess.run(['c++','-O3','-std=c++17',str(SRC/'deep-sim.cpp'),'-o',str(binary)],check=True)
        for folder,mode in cases:
            status=json.loads((folder/f'{mode}-status.json').read_text())
            for name in ['deep-sim.cpp','resources.hpp','cache.hpp']:
                assert hashlib.sha256((SRC/name).read_bytes()).hexdigest()==status['source_sha256'][name]
            for old,digest in status.get('input_sha256',{}).items():
                assert hashlib.sha256(Path(relocate(old)).read_bytes()).hexdigest()==digest
            configs=json.loads((folder/f'{mode}-configs.json').read_text())
            for c in configs:
                for key in ['trace','phases','shapes','cpu_file']:c[key]=relocate(c[key])
                c['detail']=0
            configfile=tmp/f'{mode}.tsv';output=tmp/f'{mode}-results.tsv'
            configfile.write_text(''.join('\t'.join(str(c[k]) for k in ORDER)+'\n' for c in configs))
            subprocess.run([str(binary),str(configfile),str(output),str(details)],check=True,text=True,capture_output=True)
            expected=list(csv.DictReader((folder/f'{mode}-results.tsv').open(),delimiter='\t'))
            actual=list(csv.DictReader(output.open(),delimiter='\t'));assert len(actual)==len(expected)
            maximum=0
            for a,b in zip(actual,expected):
                assert a.keys()==b.keys()
                for key in a:
                    if key=='trace':continue
                    if key in ['id','card']:assert a[key]==b[key];continue
                    av,bv=float(a[key]),float(b[key]);maximum=max(maximum,abs(av-bv))
                    assert math.isclose(av,bv,rel_tol=1e-7,abs_tol=1e-6),(mode,a['id'],key,av,bv)
            report.append({'mode':mode,'cases':len(configs),'summary_rows':len(actual),'max_absolute_numeric_difference':maximum,'passed':True})
            print(mode,len(configs),'cases: matched',flush=True)
    destination=OUT/('REPLAY_SMOKE_CHECK.json' if args.smoke else 'REPLAY_ALL_CHECK.json')
    destination.write_text(json.dumps({'source_root':str(SRC),'no_hardware_benchmarks':True,'runs':report},indent=2)+'\n')

if __name__=='__main__':main()
