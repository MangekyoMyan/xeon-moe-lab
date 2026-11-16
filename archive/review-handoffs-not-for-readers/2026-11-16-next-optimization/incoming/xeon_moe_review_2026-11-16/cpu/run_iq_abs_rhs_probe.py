#!/usr/bin/env python3
"""Build/test the input-abs idea; optionally run a small EPYC-only paired probe.

test creates a new directory; bench uses only that freshly built, hashed probe.
No model, GPU, service, shipped ELF, or tracked source is changed or executed.
"""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,timeout=90):
    r=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
    return {'argv':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=['test','bench'],default='test')
    ap.add_argument('--focused',action='store_true',help='Benchmark only K5120/8192 rows for all three IQ types')
    a=ap.parse_args()
    root=a.snapshot.resolve();out=a.output.resolve();result_path=out/'RESULT.json'
    if a.phase=='test':
        out.mkdir(parents=True,exist_ok=False)
        e=root/'snapshot/llm-lab/exploration-20261112'
        g=root/'snapshot/llm-lab/mxfp4-hybrid-20261102/hybrid/engine/ggml'
        local=Path(__file__).with_name('iq_abs_rhs_probe.cpp').resolve()
        sources=[local,e/'row-local/compact-kernel.inc',e/'row-local/iq-small-kernels.inc',e/'iq-rhs/iq-rhs.cpp',e/'iq-rhs/iq-rhs.h',e/'iq-rhs/iq-math.inc',e/'iq-rhs/compact-kernel.inc']
        include=['-I'+str(p) for p in [g/'include',g/'src',g/'src/ggml-cpu',e/'row-local',e/'iq-rhs']]
        flags=['-O3','-std=gnu++17','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off']
        compile_cmd=['c++',*flags,*include,str(local),str(e/'iq-rhs/iq-rhs.cpp'),'-lm','-o',str(out/'probe')]
        records=[run(compile_cmd)]
        if records[-1]['exit_code']==0:records.append(run([str(out/'probe'),'test']))
        result={'scope':'Synthetic standalone current-compact versus experimental abs-input and saved bias62 input-sidecar; no engine backend or model',
          'platform':platform.platform(),'cpu':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
          'source_sha256':{str(p.relative_to(root)) if p.is_relative_to(root) else str(p):sha(p) for p in sources},
          'commands':records,'test_passed':len(records)==2 and records[-1]['exit_code']==0}
        if result['test_passed']:
            result['probe_sha256']=sha(out/'probe')
            result['correctness_rows']=[json.loads(line) for line in records[-1]['stdout'].splitlines()]
            result['totals']={key:sum(r[key] for r in result['correctness_rows']) for key in ['bit_exact_outputs','unsafe_rejections','stored_bias_variant_bit_checks']}
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    else:
        result=json.loads(result_path.read_text())
        if result.get('test_passed') is not True:
            raise RuntimeError('The recorded correctness test did not pass; benchmark refused')
        if not (out/'probe').is_file() or sha(out/'probe')!=result.get('probe_sha256'):
            raise RuntimeError('The newly built probe hash changed; benchmark refused')
        bench=run([str(out/'probe'),'bench_focused' if a.focused else 'bench'],timeout=120)
        result['commands'].append(bench);result['benchmark_passed']=bench['exit_code']==0
        if result['benchmark_passed']:result['benchmark_rows']=[json.loads(line) for line in bench['stdout'].splitlines()]
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'result':str(result_path),'test_passed':result['test_passed'],'totals':result.get('totals'),
      'benchmark_passed':result.get('benchmark_passed'),'last_stderr':result['commands'][-1]['stderr'][:2000]},ensure_ascii=False))
    return 0 if result['test_passed'] and result.get('benchmark_passed',True) else 1
if __name__=='__main__':raise SystemExit(main())
