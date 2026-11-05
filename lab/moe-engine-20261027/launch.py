#!/usr/bin/env python3
"""Run the accepted Q8 hybrid backend with a validated model-bound residency plan.

Uses frozen v2 numerical kernels. Experimental kernels are kept in variants/ and
are not selected here. Model-specific measured thread choices live in profiles/.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
LAB=R.parent
FROZEN=LAB/'moe-implementation-20261026/release'
sys.path.insert(0,str(R/'tools'))
from make_plan import validate_plan


def memory_kib():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))


def prepare(args):
    baseline=json.loads((FROZEN/'READY.json').read_text())
    for path,expected in baseline['sha256'].items():
        with open(path,'rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
                raise ValueError('Accepted backend file changed: '+path)
    model=args.model or Path(baseline['model'])
    if args.plan is None and args.model is not None:
        raise ValueError('An explicit model needs its own --plan; use tools/make_plan.py to build it')
    plan=args.plan or R/'plans/glm-air-q8-22g-legacy.txt'
    inventory,manifest=validate_plan(model,plan)
    layers=inventory['metadata'].get(inventory['architecture']+'.block_count')
    if type(layers) is not int or not 0<layers<=4096:
        raise ValueError('Missing or unsupported trunk block count')
    threads=batch_threads=12
    profile_path=R/'profiles'/(inventory['model_snapshot_id']+'.json')
    profile_status='no measured profile; using conservative12threads'
    if profile_path.is_file():
        profile=json.loads(profile_path.read_text())
        if (profile.get('model_snapshot_id')==inventory['model_snapshot_id'] and
            profile.get('plan_sha256')==manifest['plan_sha256'] and
            profile.get('memory_total_kib')==memory_kib() and
            profile.get('status')=='validated'):
            threads=profile['settings']['decode_threads'];batch_threads=profile['settings']['prefill_threads']
            profile_status='measured model profile: '+str(profile_path)
        else:
            profile_status='profile not applicable to current model/plan/RAM; using12threads'
    threads=args.threads or threads;batch_threads=args.batch_threads or batch_threads
    if not 1<=threads<=24 or not 1<=batch_threads<=24:
        raise ValueError('Thread counts must fit this server (1..24)')
    env={
        'CUDA_VISIBLE_DEVICES':'GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3',
        'GGML_XV_PLUGIN':str(FROZEN/'runtime/libxv-experiment.so'),
        'GGML_XV_PLAN':str(plan.resolve()),
        'GGML_XV_CACHE_MIB':str((manifest['budget_bytes']+2**20-1)//2**20),
        'GGML_XV_WORKER_CPU':'18','GGML_XV_COPY_CPUS':'0,1,6,7',
        'GGML_XV_WHOLE_MLP':'1','GGML_XV_GATE_UP_ONLY':'1','GGML_XV_MLP_CPU_PAIR':'1',
        'GGML_XV_PINNED_ACTIVATIONS':'1','GGML_XV_PREFILL_TOKEN_AXIS':'0','GGML_CPU_Q8_MOE_ROWS4':'0'}
    label='model-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=LAB/'results'/R.name/label
    command=['env','LD_LIBRARY_PATH='+str(FROZEN/'runtime')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),
        sys.executable,str(FROZEN/'without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',
        str(FROZEN/'bin/llama-completion'),'-m',str(model.resolve()),'-t',str(threads),'-tb',str(batch_threads),
        '--poll','0','--poll-batch','0','--cpu-mask','0xffffff','--cpu-strict','1',
        '--cpu-mask-batch','0xffffff','--cpu-strict-batch','1','-c',str(args.context),'-b','128','-ub','128',
        '-n',str(args.max_tokens),'--load-mode','mmap','--lazy-mode','off','--no-repack','-ngl','all',
        '--n-cpu-moe',str(layers),'--no-op-offload','-fa','auto','--fit','off','--offline','--jinja',
        '--reasoning','off','--temp','0','--seed','42','--single-turn','--simple-io','--color','off',
        '--no-display-prompt','--perf']
    if args.prompt is not None:command+=['-p',args.prompt]
    if args.prompt_file is not None:command+=['-f',str(args.prompt_file.resolve())]
    guard=[sys.executable,str(R/'watch.py'),'--label',label,'--timeout',str(args.timeout),
           '--power-limit',str(args.power_limit)]
    for key,value in env.items():guard+=['--env',key+'='+value]
    guard+=['--',*command]
    return guard,out,{'model':str(model.resolve()),'model_snapshot_id':inventory['model_snapshot_id'],
        'plan':str(plan.resolve()),'backend':'frozen validated v2',
        'decode_threads':threads,'prefill_threads':batch_threads,'thread_profile':profile_status,
        'logs':str(out),'command':guard}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--plan',type=Path)
    prompts=parser.add_mutually_exclusive_group()
    prompts.add_argument('--prompt','-p');prompts.add_argument('--prompt-file','-f',type=Path)
    parser.add_argument('--context',type=int,default=4096)
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--threads',type=int,choices=range(1,25))
    parser.add_argument('--batch-threads',type=int,choices=range(1,25))
    parser.add_argument('--power-limit',type=int,choices=[150,175,200],default=150)
    parser.add_argument('--timeout',type=int,default=3600)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 1<=args.context<=4096 or not 1<=args.max_tokens<=args.context:parser.error('Use1<=tokens<=context<=4096')
    if not 60<=args.timeout<=14400:parser.error('Timeout must be60..14400seconds')
    if not args.dry_run and args.prompt is None and args.prompt_file is None:parser.error('Provide a prompt')
    if args.prompt_file is not None and not args.prompt_file.is_file():parser.error('Prompt file is missing')
    try:guard,out,info=prepare(args)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))
    if args.dry_run:
        print(json.dumps(info,indent=2));return
    if (R/'STOP_REQUESTED').exists():parser.error('Inspect prior stop at '+str(R/'STOP_REQUESTED'))
    busy=subprocess.check_output(['nvidia-smi','-i','GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3',
                                 '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:parser.error('GPU is in use; current run must finish first')
    prefix=[] if os.geteuid()==0 else ['sudo','-n']
    print('Backend frozenv2; threads decode='+str(info['decode_threads'])+' prefill='+str(info['prefill_threads']),file=sys.stderr)
    watcher=subprocess.Popen(prefix+guard,stdout=sys.stderr,stderr=sys.stderr)
    offset=0
    def stream():
        nonlocal offset
        path=out/'stdout.txt'
        if path.exists():
            with path.open('rb') as handle:handle.seek(offset);chunk=handle.read();offset=handle.tell()
            if chunk:sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
    try:
        while watcher.poll() is None:stream();time.sleep(.1)
    except KeyboardInterrupt:
        try:watcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(prefix+['kill','-INT',str(watcher.pid)],check=False);watcher.wait(timeout=30)
    stream()
    if out.exists():(out/'launch.json').write_text(json.dumps(info,indent=2)+'\n')
    print('\nLogs: '+str(out),file=sys.stderr)
    if watcher.returncode and (out/'stderr.txt').exists():print((out/'stderr.txt').read_text(errors='replace')[-5000:],file=sys.stderr)
    sys.exit(watcher.returncode)


if __name__=='__main__':main()
