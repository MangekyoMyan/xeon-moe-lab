#!/usr/bin/env python3
"""Run the validated Xeon/Volta v2 profile with a bounded hardware watchdog."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
L=R.parent
P=R/'release'
ap=argparse.ArgumentParser(description='GLM-4.5-Air Q8 optimized for this dual-Xeon/PG500 server')
inputs=ap.add_mutually_exclusive_group()
inputs.add_argument('--prompt','-p')
inputs.add_argument('--prompt-file','-f',type=Path)
ap.add_argument('--context',type=int,default=4096)
ap.add_argument('--max-tokens',type=int,default=256)
ap.add_argument('--power-limit',type=int,choices=[150,175,200],default=150)
ap.add_argument('--timeout',type=int,default=3600)
ap.add_argument('--dry-run',action='store_true')
a=ap.parse_args()
if not 1<=a.context<=4096 or not 1<=a.max_tokens<=a.context:ap.error('Use context<=4096 and a positive token limit within it.')
if not 60<=a.timeout<=14400:ap.error('Use timeout between60 and14400seconds.')
if not a.dry_run and a.prompt is None and a.prompt_file is None:ap.error('Provide --prompt or --prompt-file.')
if a.prompt_file is not None and not a.prompt_file.is_file():ap.error('Prompt file does not exist.')
ready=P/'READY.json'
if not ready.is_file():ap.error('The v2 release has not finished validation.')
manifest=json.loads(ready.read_text())
for path,expected in manifest['sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:ap.error('Validated file changed: '+path)
for path,size in manifest['model_shards'].items():
    if not Path(path).is_file() or Path(path).stat().st_size!=size:ap.error('Expected retained model is missing or changed: '+path)
model=manifest['model']
label='v2-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out=L/'results'/R.name/label
env={
    'CUDA_VISIBLE_DEVICES':'GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3',
    'GGML_XV_PLUGIN':str(P/'runtime/libxv-experiment.so'),
    'GGML_XV_PLAN':str(P/'global-22g.txt'),
    'GGML_XV_WORKER_CPU':'18','GGML_XV_COPY_CPUS':'0,1,6,7',
    'GGML_XV_WHOLE_MLP':'1','GGML_XV_GATE_UP_ONLY':'1','GGML_XV_MLP_CPU_PAIR':'1',
    'GGML_XV_PINNED_ACTIVATIONS':'1','GGML_XV_PREFILL_TOKEN_AXIS':'0','GGML_CPU_Q8_MOE_ROWS4':'0',
}
command=['env','LD_LIBRARY_PATH='+str(P/'runtime')+':'+str(L/'v100-prep-20261024/cuda-12.9.2/lib'),
    sys.executable,str(P/'without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',
    str(P/'bin/llama-completion'),'-m',model,'-t','12','-tb','12','--poll','0','--poll-batch','0',
    '--cpu-mask','0xffffff','--cpu-strict','1','--cpu-mask-batch','0xffffff','--cpu-strict-batch','1',
    '-c',str(a.context),'-b','128','-ub','128','-n',str(a.max_tokens),'--load-mode','mmap',
    '--lazy-mode','off','--no-repack','-ngl','all','--n-cpu-moe','46','--no-op-offload',
    '-fa','auto','--fit','off','--offline','--jinja','--reasoning','off','--temp','0','--seed','42',
    '--single-turn','--simple-io','--color','off','--no-display-prompt','--perf']
if a.prompt is not None:command+=['-p',a.prompt]
if a.prompt_file is not None:command+=['-f',str(a.prompt_file.resolve())]
guard=[sys.executable,str(R/'watch.py'),'--label',label,'--timeout',str(a.timeout),'--power-limit',str(a.power_limit)]
for k,v in env.items():guard+=['--env',k+'='+v]
guard+=['--',*command]
if a.dry_run:
    print(json.dumps({'command':guard,'logs':str(out),'profile':manifest['profile'],'restored_power_W':150},indent=2));sys.exit(0)
if (R/'STOP_REQUESTED').exists():ap.error('A previous monitored run stopped; inspect '+str(R/'STOP_REQUESTED')+' before starting another.')
# Catch older launchers or external CUDA programs that do not use our lock.
busy=subprocess.check_output(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if busy:ap.error('GPU is already in use; let the current run finish.')
prefix=[] if os.geteuid()==0 else ['sudo','-n']
watcher=subprocess.Popen(prefix+guard,stdout=sys.stderr,stderr=sys.stderr)
offset=0
def stream_new_output():
    global offset
    file=out/'stdout.txt'
    if file.exists():
        with file.open('rb') as f:
            f.seek(offset);chunk=f.read();offset=f.tell()
        if chunk:sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
try:
    while watcher.poll() is None:
        stream_new_output();time.sleep(.1)
except KeyboardInterrupt:
    # The foreground watchdog also receives terminal SIGINT and restores150W.
    # Forward an interrupt when this launcher alone was interrupted programmatically.
    try:watcher.wait(timeout=5)
    except subprocess.TimeoutExpired:
        subprocess.run(prefix+['kill','-INT',str(watcher.pid)],check=False)
        watcher.wait(timeout=30)
stream_new_output()
print('\nLogs: '+str(out),file=sys.stderr)
if watcher.returncode:
    file=out/'stderr.txt'
    if file.exists():print(file.read_text(errors='replace')[-5000:],file=sys.stderr)
sys.exit(watcher.returncode)
