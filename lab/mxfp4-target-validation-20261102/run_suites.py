#!/usr/bin/env python3
"""Sequential CPU-only A/B suites. Production API remains stopped."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT.parent/'pro-review-received-20261102/deepseek-cpu-review-result'
BIN={v:ROOT/v/'build-cpu-ivybridge/bin/cpu-quant-probe' for v in ['stock','mxfp4']}

def verify_binaries():
    records={}
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);env.pop('LD_PRELOAD',None)
    for variant,binary in BIN.items():
        output=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
        libs={}
        for line in output.splitlines():
            if 'libggml-' in line:
                name,path=line.split('=>',1);path=Path(path.strip().split()[0]).resolve()
                assert path.parent==binary.parent.resolve(),(variant,path)
                libs[name.strip()]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        assert set(libs)=={'libggml-cpu.so.0','libggml-base.so.0'},libs
        symbols=subprocess.check_output(['nm','-D',str(binary.parent/'libggml-cpu.so.0')],text=True)
        present='ggml_vec_dot_mxfp4_q8_0_cols2' in symbols
        assert present==(variant=='mxfp4')
        records[variant]={'binary':str(binary),'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                          'libraries':libs,'paired_kernel_exported':present,'ldd':output}
    assert records['stock']['libraries']['libggml-cpu.so.0']['sha256']!=records['mxfp4']['libraries']['libggml-cpu.so.0']['sha256']
    (ROOT/'binary-provenance.json').write_text(json.dumps(records,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['correctness','performance','pinned','routing'],required=True);a=ap.parse_args()
    verify_binaries()
    lock=(ROOT.parent/'results/glm-air-q8-20261020/.experiment.lock').open('rb')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suites=[]
    if a.phase=='correctness':
        for threads,cpus in [(1,'0'),(24,'0-23')]:
            suites.append((f'boundary-t{threads}',cpus,['--interleave=0,1'],
              ['--type','all','--mode','both','--k','512','--m','259','--tokens','1,2,3,7,8,9,16,17',
               '--experts','4','--topk','2','--routing','shared','--threads',str(threads),'--input-pad','7','--repeats','3'],300))
    elif a.phase in ['performance','pinned']:
        placements=[(1,'0','t1'),(6,'0-2,6-8','t6-balanced'),(12,'0-11','t12-physical'),(24,'0-23','t24-smt')]
        if a.phase=='pinned':placements=placements[1:]
        for threads,cpus,label in placements:
            suites.append((f'{"pinned" if a.phase=="pinned" else "real"}-{label}',cpus,['--interleave=0,1'],
              ['--type','mxfp4','--real-shapes','--mode','both','--experts','6','--topk','6','--routing','shared',
               '--tokens','1,2,3,8,17','--threads',str(threads),'--repeats',('21' if a.phase=='pinned' else '11'),
               '--warmups',('5' if a.phase=='pinned' else '3')]+(['--strict-cpu'] if a.phase=='pinned' else []),600))
    else:
        for threads,cpus in [(12,'0-11'),(24,'0-23')]:
            suites.append((f'spread-real-t{threads}',cpus,['--interleave=0,1'],
              ['--type','mxfp4','--real-shapes','--mode','both','--experts','12','--topk','6','--routing','spread',
               '--tokens','1,2,3,6,8,17','--threads',str(threads),'--repeats','11','--warmups','3','--strict-cpu'],600))
        suites.append(('spread384-small-t24','0-23',['--interleave=0,1'],
          ['--type','mxfp4','--mode','both','--k','256','--m','33','--experts','384','--topk','6','--routing','spread',
           '--tokens','1,8,128','--threads','24','--repeats','5','--warmups','2','--strict-cpu'],300))
    state={'phase':a.phase,'started_unix':time.time(),'completed':[],'status':'running'}
    state_path=ROOT/('phase-'+a.phase+'.json')
    try:
        for name,cpus,numa,probe,timeout in suites:
            out=ROOT/'results'/name
            runner=['numactl',*numa,'python3',str(BUNDLE/'probe/run_compare.py'),
                    '--stock',str(BIN['stock']),'--candidate',str(BIN['mxfp4']),
                    '--out',str(out),'--rounds',('4' if a.phase=='pinned' else '2'),'--cpus',cpus,'--',*probe]
            state['current']=name;state_path.write_text(json.dumps(state,indent=2)+'\n')
            subprocess.run(['python3',str(ROOT/'guarded_run.py'),'--name',name,'--timeout',str(timeout),'--',*runner],check=True)
            manifest=json.loads((out/'manifest.json').read_text());assert manifest['passed']
            (out/'placement.json').write_text(json.dumps({'outer_numactl':numa,'affinity':cpus,
                'placement_applied':True,'numa_policy_inherited_by_children':True,'clock_control':False,
                'note':'runner internal numa_control=false refers to its own controls; numactl applied externally'},indent=2)+'\n')
            rows=[json.loads(x) for x in (out/'summary.jsonl').read_text().splitlines()]
            summary={'suite':name,'cases':len(rows),'speed_ratio_min':min(x['speed_ratio'] for x in rows),
                     'speed_ratio_max':max(x['speed_ratio'] for x in rows),'all_outputs_bit_exact':True}
            state['completed'].append(summary);print(json.dumps(summary),flush=True)
        state['status']='passed'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        state['finished_unix']=time.time();state_path.write_text(json.dumps(state,indent=2)+'\n');lock.close()

if __name__=='__main__':main()
