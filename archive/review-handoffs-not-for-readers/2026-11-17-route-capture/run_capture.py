"""User-run diagnostic replacement window; never stops/restarts the ordinary API.

--prepare validates code/profile without loading a model. Default run requires the
API already stopped and uses the existing supervisor unchanged except local output
directory, a64GiB RAM reserve and a2hour overall budget. No hardware setting writes.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,resource,subprocess,sys,time,uuid
R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
CONFIG_SHA='963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2)+'\n')
def prepare(continue64=False):
    cfgpath=ENGINE/'server-config.json';assert sha(cfgpath)==CONFIG_SHA,'Current API profile changed; re-review first'
    cfg=json.loads(cfgpath.read_text());link=json.loads((R/'LINK_BUILD.json').read_text());clients=json.loads((R/'CLIENT_BUILD.json').read_text())
    assert sha(R/'runtime/libllama.so.0.8.0')==link['diagnostic_llama_sha256']
    assert sha(R/'driver')==clients['binaries']['driver']
    for v in link['runtime_libraries'].values():assert sha(Path(v['path']))==v['sha256']
    for name,digest in clients['source_sha256'].items():assert sha(R/name)==digest
    normal=cfg['command'];server=str(ENGINE/'build/bin/llama-server');idx=normal.index(server)
    prefix=normal[:idx]
    assert prefix==['/usr/bin/python3',str(LAB/'moe-implementation-20261026/release/without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1']
    values={'-m','-t','-tb','--poll','--poll-batch','--cpu-mask','--cpu-strict','--cpu-mask-batch','--cpu-strict-batch','-c','-b','-ub','--load-mode','--lazy-mode','-ngl','--n-cpu-moe','--override-tensor','-fa','--cache-type-k','--cache-type-v','--fit','--parallel','--mmproj','--image-max-tokens','--mtmd-batch-max-tokens'}
    flags={'--no-repack','--op-offload','--offline','--no-context-shift','--mmproj-offload'}
    args=[];i=idx+1
    while i<len(normal):
        token=normal[i]
        if token in values:args.extend(normal[i:i+2]);i+=2
        elif token in flags:args.append(token);i+=1
        else:i+=1
    for flag,value in [('-c','131072'),('--parallel','1'),('--n-cpu-moe','37'),('--fit','off')]:assert args[args.index(flag)+1]==value
    model=Path(args[args.index('-m')+1]);assert model.is_file()
    model_root=model.parent.parent;manifest=json.loads((model_root/'download-verified.json').read_text())
    for item in manifest['files']:assert (model_root/item['name']).stat().st_size==item['size']
    env={k:v for k,v in cfg['environment'].items() if k.startswith(('GGML_','OMP_','CUDA_','LD_'))}
    env['LD_LIBRARY_PATH']=str(R/'runtime')+':'+env['LD_LIBRARY_PATH'];env['LD_BIND_NOW']='1'
    assert env['GGML_CPU_IQ3_COMPACT_MIB']=='122880' and env['GGML_OP_OFFLOAD_MIN_BATCH']=='64' and env['GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH']=='64'
    plan=Path(env['GGML_CPU_IQ_PLAN']);assert sha(plan)=='cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6'
    return {'prefix':prefix,'driver':str(R/'driver'),'args':args,'environment':env,'continue64':continue64,'planned_calls':68 if continue64 else 4,'baseline_config_sha256':CONFIG_SHA,'driver_sha256':sha(R/'driver'),'diagnostic_llama_sha256':sha(R/'runtime/libllama.so.0.8.0'),'guard_source_sha256':{str(p):sha(p) for p in [ENGINE/'supervise.py',ENGINE/'gpu_guard.py',LAB/'moe-engine-20261029/watch.py']},'RAM_reserve_GiB':64,'API_automatically_stopped_or_restarted':False,'model_manifest_sha256':sha(model_root/'download-verified.json')}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare',action='store_true');ap.add_argument('--continue64',action='store_true');a=ap.parse_args();plan=prepare(a.continue64)
    if a.prepare:write(R/'LAUNCH_PLAN.json',plan);print(json.dumps({'prepared':True,'model_started':False,'planned_calls':plan['planned_calls'],'file':str(R/'LAUNCH_PLAN.json')}));return 0
    if os.geteuid()!=0:raise RuntimeError('Run with sudo; supervisor must read existing thermal/fan controls')
    for name in ['llm-deepseek41-api.service','llm-openai-api.service']:
        if subprocess.run(['systemctl','is-active','--quiet',name]).returncode==0:raise RuntimeError(name+' is running; not stopping it automatically')
    for name in ['xeon-watch.service','xeon-watch-notify.service']:
        if subprocess.run(['systemctl','is-active','--quiet',name]).returncode:raise RuntimeError('Monitor service inactive: '+name)
    for p in [ENGINE/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED']:
        if p.exists():raise RuntimeError('Existing hardware stop must be inspected: '+str(p))
    prior_stops=list((R/'runs').glob('*/STOP_REQUESTED'))
    if prior_stops:raise RuntimeError('Previous diagnostic protection stop must be inspected: '+str(prior_stops[0]))
    snap=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());assert 0<=time.time()-snap['unix']<=45 and not snap['errors'] and not snap['active_alerts'],'monitor unavailable/alert'
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    assert mem['MemTotal']>=480*1024**3 and mem['MemAvailable']>=360*1024**3 and mem['Mlocked']<=32*1024**3,'Full model launch requires released ordinary context and available RAM'
    owner=LAB.stat();root=R/'runs';root.mkdir(exist_ok=True);run=root/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'Z-'+uuid.uuid4().hex[:8]);run.mkdir();os.chown(run,owner.st_uid,owner.st_gid)
    command=[*plan['prefix'],plan['driver'],'--route-output',str(run),*(['--route-continue64'] if a.continue64 else []),*plan['args']]
    write(run/'server-config.json',{'command':command,'environment':plan['environment']});write(run/'LAUNCH_MANIFEST.json',plan)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY))
    sys.path.insert(0,str(ENGINE));spec=importlib.util.spec_from_file_location('route37_existing_supervisor',ENGINE/'supervise.py');supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    supervisor.ROOT=run
    original=supervisor.check_host;deadline=time.monotonic()+7200
    def strict_host(initial):
        cpu,available=original(initial)
        if available<64*1024**2:raise RuntimeError('Diagnostic RAM reserve below64GiB')
        if time.monotonic()>deadline:raise TimeoutError('Diagnostic2hour budget exceeded')
        return cpu,available
    supervisor.check_host=strict_host
    print('診断出力: '+str(run),flush=True);print('進捗: sudo tail -f '+str(run/'server.stderr.log'),flush=True)
    code=supervisor.main()
    if (run/'RUN.json').exists():
        result=json.loads((run/'RUN.json').read_text());code=code or (0 if result.get('passed') else 1)
    else:code=code or 1
    print('終了: '+str(code)+' 結果: '+str(run/'RUN.json'),flush=True)
    if (run/'STOP_REQUESTED').exists():print('保護停止の理由を確認してください。通常APIも自動再起動しません: '+str(run/'STOP_REQUESTED'),flush=True)
    else:print('通常APIは自動再起動しません。復帰: sudo python3 '+str(ENGINE/'start.py'),flush=True)
    return code
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:raise SystemExit(str(error))
