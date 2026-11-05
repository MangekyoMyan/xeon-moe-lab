#!/usr/bin/env python3
"""Validated CPU/GPU hybrid Q8 inference with reusable decode work buffers."""
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
VARIANT=R/'variants/hybrid-reuse-v1'
RUNTIME=VARIANT/'runtime'
sys.path.insert(0,str(R/'tools'))
from make_plan import validate_plan


def memory_kib():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')))


def cpu_topology():
    info=Path('/proc/cpuinfo').read_text().splitlines()
    model=next(line.split(':',1)[1].strip() for line in info if line.startswith('model name'))
    return {'model':model,'online':Path('/sys/devices/system/cpu/online').read_text().strip(),
            'numa_cpus':{p.name:(p/'cpulist').read_text().strip()
                         for p in sorted(Path('/sys/devices/system/node').glob('node[0-9]*'))}}


def profile_hardware_matches(profile):
    # MemTotal changed by four KiB after a reboot on this unchanged server.
    # Permit only small reservation differences; real RAM upgrades invalidate it.
    previous=profile.get('memory_total_kib')
    return (type(previous) is int and abs(previous-memory_kib())<=64 and
            profile.get('cpu_topology')==cpu_topology())


def prepare(args):
    variant=VARIANT
    runtime=RUNTIME
    if getattr(args,'gpu_kernel',None)=='warp32':
        variant=R/'variants/hybrid-reuse-v1-warp32';runtime=variant/'runtime'
    baseline=json.loads((FROZEN/'READY.json').read_text())
    for path,expected in baseline['sha256'].items():
        with open(path,'rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
                raise ValueError('Accepted backend file changed: '+path)
    build=json.loads((variant/'BUILD.json').read_text())
    for path,expected in build['sha256'].items():
        with open(path,'rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=expected:
                raise ValueError('Candidate runtime or pinned CUDA ABI changed: '+path)
    model=args.model or Path(baseline['model'])
    if args.plan is None and args.model is not None:
        raise ValueError('An explicit model needs its own --plan; use tools/make_plan.py to build it')
    plan=args.plan or R/'plans/glm-air-q8-22g-legacy.txt'
    inventory,manifest=validate_plan(model,plan)
    prior_path=None;prior_manifest=None;adaptive=not args.static
    prior_status='explicit static residency' if args.static else 'model-bound prior'
    if adaptive:
        prior_path=args.prior or R/'plans/glm-air-q8-prior.txt'
        prior_manifest=json.loads(prior_path.with_suffix('.json').read_text())
        if prior_manifest.get('schema_version')!=1:
            raise ValueError('Unsupported adaptive prior manifest')
        if args.prior is None and prior_manifest.get('model_snapshot_id')!=inventory['model_snapshot_id']:
            # Keep the existing --model/--plan interface usable for a different
            # model. Never borrow another model's observed expert frequencies.
            adaptive=False;prior_path=None;prior_manifest=None
            prior_status='no matching automatic prior; static residency'
        elif prior_manifest.get('model_snapshot_id')!=inventory['model_snapshot_id']:
            raise ValueError('Adaptive prior belongs to a different model')
        if adaptive and hashlib.sha256(prior_path.read_bytes()).hexdigest()!=prior_manifest['prior_sha256']:
            raise ValueError('Adaptive prior changed')
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
            profile_hardware_matches(profile) and
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
        'GGML_XV_PLUGIN':str(runtime/'libxv-experiment.so'),
        'GGML_XV_EXACT_Q8':'1',
        'GGML_XV_PLAN':str(plan.resolve()),
        'GGML_XV_CACHE_MIB':str((manifest['budget_bytes']+2**20-1)//2**20),
        'GGML_XV_WORKER_CPU':'18','GGML_XV_COPY_CPUS':'0,1,6,7',
        'GGML_XV_WHOLE_MLP':'1','GGML_XV_GATE_UP_ONLY':'1','GGML_XV_MLP_CPU_PAIR':'1',
        'GGML_XV_PINNED_ACTIVATIONS':'1','GGML_XV_PREFILL_TOKEN_AXIS':'0','GGML_CPU_Q8_MOE_ROWS4':'0'}
    share_graphs=bool(build.get('features',{}).get('share_decode_graphs',False))
    if getattr(args,'no_graph_reuse',False):share_graphs=False
    env['GGML_XV_SHARE_DECODE_GRAPHS']=str(int(share_graphs))
    if adaptive:
        env.update(GGML_XV_ADAPTIVE='1',GGML_XV_ADAPTIVE_PRIOR=str(prior_path.resolve()),
                   GGML_XV_MODEL_SNAPSHOT=inventory['model_snapshot_id'],GGML_XV_ADAPTIVE_PIPELINE='1',
                   GGML_XV_AUTO_STEP='1',GGML_XV_DEFER_COMMIT='1')
    label='model-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=LAB/'results'/R.name/label
    command=['env','LD_LIBRARY_PATH='+str(runtime)+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib'),
        sys.executable,str(FROZEN/'without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',
        'env','LD_PRELOAD='+(str(runtime/'libxv-auto-step.so') if adaptive else ''),
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
        'plan':str(plan.resolve()),'backend':'CPU-canonical hybrid Q8 with '+('adaptive' if adaptive else 'static')+' residency',
        'variant':str(variant),'gpu_kernel':build.get('features',{}).get('gpu_kernel','warp32' if variant.name=='exact-v5' else 'baseline'),
        'decode_graph_reuse':share_graphs,'adaptive':adaptive,'prior_status':prior_status,
        'deferred_cache_updates':adaptive,
        'prior_path':str(prior_path.resolve()) if prior_path else None,'prior_sha256':prior_manifest['prior_sha256'] if prior_manifest else None,
        'decode_threads':threads,'prefill_threads':batch_threads,'thread_profile':profile_status,
        'logs':str(out),'command':guard}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--prior',type=Path)
    parser.add_argument('--static',action='store_true',help='Keep exact GPU arithmetic but do not change residency')
    parser.add_argument('--no-graph-reuse',action='store_true',help='Disable shared decode buffers for comparison or rollback')
    parser.add_argument('--gpu-kernel',choices=['baseline','warp32'],default='baseline',
                        help='warp32 is bit-exact and faster in kernel tests; whole-model benefit is not established')
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
    if info['prior_path']:
        raw=Path(info['prior_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=info['prior_sha256']:parser.error('Prior changed before launch')
        folder=R/'prior-snapshots';folder.mkdir(exist_ok=True)
        snapshot=folder/(info['prior_sha256']+'.txt')
        if not snapshot.exists():snapshot.write_bytes(raw);snapshot.chmod(0o444)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=info['prior_sha256']:parser.error('Prior snapshot changed')
        for i,value in enumerate(guard):
            if value=='GGML_XV_ADAPTIVE_PRIOR='+info['prior_path']:guard[i]='GGML_XV_ADAPTIVE_PRIOR='+str(snapshot)
        info['prior_snapshot']=str(snapshot)
    if (R/'STOP_REQUESTED').exists():parser.error('Inspect prior stop at '+str(R/'STOP_REQUESTED'))
    busy=subprocess.check_output(['nvidia-smi','-i','GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3',
                                 '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy:parser.error('GPU is in use; current run must finish first')
    prefix=[] if os.geteuid()==0 else ['sudo','-n']
    print(info['backend']+'; threads decode='+str(info['decode_threads'])+' prefill='+str(info['prefill_threads']),file=sys.stderr)
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
