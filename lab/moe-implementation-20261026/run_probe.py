#!/usr/bin/env python3
"""One fixed-input model run. Run only inside watch.py after preload finishes."""
import argparse, hashlib, json, os, subprocess, resource, time, re
from pathlib import Path
R=Path(__file__).resolve().parent
E=R.parent/'xeon-volta-engine'
ap=argparse.ArgumentParser()
ap.add_argument('label')
ap.add_argument('--rows',default='0',choices=['0','1','4096','1408'])
ap.add_argument('--axis',default='0',choices=['0','1'])
ap.add_argument('--original',action='store_true')
ap.add_argument('--records',default=128,type=int)
ap.add_argument('--prefix',default=63,type=int)
ap.add_argument('--text',default='heldout-text.txt')
ap.add_argument('--save-logits',action='store_true')
ap.add_argument('--measure',action='store_true')
ap.add_argument('--mlp',action='store_true')
ap.add_argument('--runtime-mlp',action='store_true')
ap.add_argument('--trace-mlp',action='store_true')
ap.add_argument('--interleave',action='store_true')
ap.add_argument('--pair',action='store_true')
ap.add_argument('--gateup',action='store_true')
ap.add_argument('--pinned',action='store_true')
ap.add_argument('--profile',action='store_true',help='Diagnostic-only core/IMC counters in a steady decode window; exclude throughput from A/B')
args=ap.parse_args()
if args.pinned: args.gateup=True
out=R/'model-runs'/args.label; out.mkdir(parents=True,exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('GGML_')}
assert not (args.original and (args.mlp or args.runtime_mlp or args.pair or args.gateup))
runtime=E/'runtime' if args.original else R/('runtime-io' if args.pinned else 'runtime-gateup' if args.gateup else 'runtime-pair' if args.pair else 'runtime-mlp' if args.mlp or args.runtime_mlp else 'runtime')
env.update(LD_LIBRARY_PATH=str(runtime),
  GGML_XV_PLUGIN=str(E/'runtime/libxv-expert-cache-parallel-copy.so' if args.original else runtime/'libxv-experiment.so'),
  GGML_XV_PLAN=str(E/'plans/global-22g.txt'),GGML_XV_WORKER_CPU='18',GGML_XV_COPY_CPUS='0,1,6,7',
  GGML_CPU_Q8_MOE_ROWS4=args.rows,GGML_XV_PREFILL_TOKEN_AXIS=args.axis)
if args.measure: env['GGML_XV_MEASURE_PHASES']='1'
if args.mlp or args.pair or args.gateup: env['GGML_XV_WHOLE_MLP']='1'
if args.pair or args.gateup: env['GGML_XV_MLP_CPU_PAIR']='1'
if args.gateup: env['GGML_XV_GATE_UP_ONLY']='1'
if args.pinned: env['GGML_XV_PINNED_ACTIVATIONS']='1'
if args.trace_mlp: env['GGML_XV_TRACE_MLP_SHAPE']='1'
exe=E/'experiments/model-probe-timed'
model=R.parent/'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf'
command=[str(exe),str(model),str(out/'logits.bin') if args.save_logits else '/dev/null',str(E/'experiments'/args.text),str(args.prefix),str(args.records)]
if args.interleave: command=['numactl','--physcpubind=0-23','--interleave=0,1',*command]
record={'argv':command,'env':{k:v for k,v in env.items() if k.startswith('GGML_') or k=='LD_LIBRARY_PATH'},'args':vars(args),
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),exe,Path(env['GGML_XV_PLUGIN']),Path(env['LD_LIBRARY_PATH'])/'libggml-cpu.so',E/'plans/global-22g.txt',E/'experiments'/args.text]}}
(out/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
    started=time.monotonic()
    child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
    record['model_pid']=child.pid
    (out/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
    profilers=[];profile_logs=[];profile={}
    if args.profile:
        while child.poll() is None:
            positions=re.findall(r'position=(\d+)',(out/'stdout.txt').read_text())
            position=int(positions[-1]) if positions else -1
            if not profilers and position>=args.prefix+31:
                specs=[('core',['-p',str(child.pid),'-e','task-clock,cycles:u,instructions:u,cycle_activity.stalls_total:u,cycle_activity.stalls_l2_pending:u,context-switches,minor-faults,major-faults']),
                       ('imc',['-a','-A','-I','1000','-e','uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/'])]
                profile={'start_position':position,'start_unix':time.time(),'commands':[],'scope':'8second diagnostic window; IMC is system-wide; intervals overlap and are not additive CPU percentages'}
                for name,opts in specs:
                    command=['sudo','-n','perf','stat','-x',',','--no-big-num','-o',str(out/(name+'.csv')),*opts,'--','sleep','8']
                    handle=(out/(name+'.stderr')).open('w');profile_logs.append(handle)
                    profile['commands'].append(command)
                    profilers.append(subprocess.Popen(command,stdout=handle,stderr=handle))
            if profilers and 'end_unix' not in profile and all(p.poll() is not None for p in profilers):
                profile['end_unix']=time.time();profile['end_position']=position
            time.sleep(.05)
    rc=child.wait(); assert rc==0,('model subprocess failed',rc)
    if args.profile:
        assert len(profilers)==2 and all(p.wait(timeout=15)==0 for p in profilers),'PMU capture failed'
        for handle in profile_logs:handle.close()
        assert profile.get('end_position',args.prefix+args.records)<args.prefix+args.records-2,'Counter window ran beyond active decode'
        (out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    wall=time.monotonic()-started
result={key:[json.loads(line[len(key)+1:]) for line in (out/'stderr.txt').read_text().splitlines() if line.startswith(key+' ')] for key in ['XV_TIMING','XV_CACHE','XV_PHASE','XV_INTERVAL','XV_LOAD','XV_MLP','XV_CPU_PAIR','XV_IO']}
assert len(result['XV_TIMING'])==1
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
result['process']={'wall_s':wall,'user_s':usage.ru_utime,'system_s':usage.ru_stime,'minor_faults':usage.ru_minflt,'major_faults':usage.ru_majflt,'maxrss_KiB':usage.ru_maxrss,'block_in':usage.ru_inblock,'block_out':usage.ru_oublock}
if args.save_logits: result['logits_sha256']=hashlib.sha256((out/'logits.bin').read_bytes()).hexdigest()
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(args.label,json.dumps(result),flush=True)
