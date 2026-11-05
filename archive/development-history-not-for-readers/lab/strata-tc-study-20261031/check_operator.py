#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;LAB=R.parent
ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['hybrid-tc','stream-tc','phase-tc','phase-v2','phase-v3']);ap.add_argument('--label');ap.add_argument('--extended',action='store_true');ap.add_argument('--boundaries',action='store_true');a=ap.parse_args()
out=R/'results'/(a.label or ('operator-'+a.variant));out.mkdir(exist_ok=False);rt=R/'variants'/a.variant/'runtime'
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','OMP_','GOMP_')) and k not in ['LD_LIBRARY_PATH','LD_PRELOAD']}
env.update(LD_LIBRARY_PATH=str(rt)+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),
 GGML_XV_PLUGIN=str(rt/'libxv-experiment.so'),GGML_XV_EXACT_Q8='1',GGML_XV_TC_PREFILL='1',GGML_XV_TC_MIN_ROWS='1',
 GGML_XV_SHARE_DECODE_GRAPHS='1',GGML_XV_WHOLE_MLP='1',GGML_XV_GATE_UP_ONLY='1',GGML_XV_MLP_CPU_PAIR='1',
 GGML_CPU_MOE_PAIR='0',GGML_XV_TEST_MUTABLE='1',GGML_XV_TEST_SLOTS='3',GGML_XV_WORKER_CPU='18',
 GGML_XV_COPY_CPUS='0,1,6,7',GGML_XV_PINNED_ACTIVATIONS='1',GGML_XV_CHECK_PINNED='1')
if a.variant=='stream-tc':env.update(GGML_XV_STREAM_ACTIVE='1',GGML_XV_STREAM_MIB='256',GGML_XV_WHOLE_MLP='0')
if a.variant in ['phase-tc','phase-v2','phase-v3']:env.update(GGML_XV_STREAM_ACTIVE='1',GGML_XV_STREAM_PREFILL_ONLY='1',GGML_XV_STREAM_MIB='256')
old=LAB/'moe-engine-20261029/experiments/hybrid-audit-agent/correctness'
oracle=old/'existing-cpu.bin';expected=next(x['sha256'] for x in json.loads((old/'results.json').read_text()) if x['suite']=='existing' and x['mode']=='cpu')
binary=LAB/'moe-engine-20261029/tests/prefill-strided-cases'
if a.extended or a.boundaries:
    binary=R/('prefill-boundaries' if a.boundaries else 'prefill-extended');oracle=out/'cpu-reference.bin'
    cpu_env=env.copy();cpu_env.pop('GGML_XV_PLUGIN',None)
    with (out/'cpu.stdout').open('w') as f,(out/'cpu.stderr').open('w') as e:
        c=subprocess.run(['numactl','--interleave=0,1',str(binary),str(oracle)],env=cpu_env,stdout=f,stderr=e)
    assert c.returncode==0
    expected=hashlib.sha256(oracle.read_bytes()).hexdigest()
assert hashlib.sha256(oracle.read_bytes()).hexdigest()==expected
with (out/'stdout.txt').open('w') as f,(out/'stderr.txt').open('w') as e:
 p=subprocess.run(['numactl','--interleave=0,1',str(binary),str(out/'output.bin')],env=env,stdout=f,stderr=e)
result={'operator_returncode':p.returncode,'oracle_sha256':expected,'variant':a.variant,'env':env.get('GGML_XV_TC_MIN_ROWS'),
        'test_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [rt/'libxv-q8-exact.so',rt/'libxv-experiment.so']}}
if p.returncode==0:
 with (out/'comparison.jsonl').open('w') as f,(out/'comparison.stderr').open('w') as e:
  c=subprocess.run([str(R/'compare-operators'),str(oracle),str(out/'output.bin')],stdout=f,stderr=e)
 result['comparison_returncode']=c.returncode
 result['summary']=(out/'comparison.stderr').read_text()
result['success']=p.returncode==0 and result.get('comparison_returncode')==0
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
