"""One frozen validation stage under the existing guard; no implicit API stop/start."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,resource,subprocess,sys,time,uuid
R=Path(__file__).resolve().parent;LAB=R.parent.parent.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
PRIOR=LAB/'parallel-pilot-20261117/integrated';GPU='GPU-6622f7a2-ee1a-a79e-1277-cee96d850aa3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def prepare(name):
    build=json.loads((R/'BUILD.json').read_text());assert build['returncode']==0 and sha(R/'driver')==build['driver_sha256']
    for n,h in build['source_sha256'].items():assert sha(R/n)==h,n
    for v in build['engine_libraries']['diagnostic'].values():assert sha(Path(v['path']))==v['sha256']
    assert sha(R/'FROZEN_PLAN.json')=='615add2b3da6befbc8c96f45fe5d09c5d14b049fd9fcf76f0a5ee8e890b07ba5'
    assert sha(R/'capacity.plan')=='4b5a8063ae1e61c55d30f957798a7cc44d7f462cf22767d2596105ed7322c596'
    frozen=json.loads((R/'FROZEN_PLAN.json').read_text());assert frozen['bank']['plan_text_exact']==(R/'capacity.plan').read_text()
    specs=[s for s in json.loads((R/'DESIGN.json').read_text())['stages'] if s['name']==name];assert len(specs)==1;spec=specs[0]
    old=module('prior_integrated_runner',PRIOR/'run_integrated.py')
    prior_build,base,args,env=old.prepare(1024,'CSP' if spec['mode']!='F' else 'F-before')
    assert build['engine_libraries']==prior_build['engine_libraries'],'unchanged engine libraries'
    env['LD_LIBRARY_PATH']=str(R/'runtime')+':'+str(ENGINE/'build/bin')+':'+str(LAB/'v100-prep-20261024/cuda-12.9.2/lib')
    env['GGML_XEON_HOT_PLAN']=str(R/'capacity.plan')
    return build,base,args,env,spec,old
def successful(run):
    h=json.loads((run/'health.json').read_text());assert h['status']=='exited' and h['child_returncode']==0
    assert json.loads((run/'RUN.json').read_text())['passed']
    m=json.loads((run/'RESULTS_SHA256.json').read_text())
    for n,h in m.items():assert sha(run/n)==h,n
    return m
def prerequisites(spec):
    # A persisted, pinned gate is required before timing/free answers, not just a previous process exit.
    if spec['stage']=='quality':return
    p=R.parent/'QUALITY_GATE.json';g=json.loads(p.read_text());assert g['passed'] and g['plan_sha256']==sha(R/'FROZEN_PLAN.json')
    expected=json.loads((R/'FROZEN_PLAN.json').read_text())['selected_case_ids_in_fixed_order']
    assert set(g['runs'])=={'F','SP'} and [c['case_id'] for c in g['cases']]==expected
    assert all(c['S_P_equal'] and c['finite'] and c['shape']==[65,129280] for c in g['cases'])
    for v in g['runs'].values():
        run=Path(v['path']);assert sha(run/'RESULTS_SHA256.json')==v['manifest_sha256'];successful(run)
    if spec['stage']=='answer':
        p=R.parent/'TIMING_COMPLETE.json';g=json.loads(p.read_text());assert g['passed'] and g['plan_sha256']==sha(R/'FROZEN_PLAN.json')
        assert len(g['runs'])==4 and g['timed_calls']==768
        for v in g['runs']:
            run=Path(v['path']);assert sha(run/'RESULTS_SHA256.json')==v['manifest_sha256'];successful(run)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('name');ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    build,base,args,env,spec,old=prepare(a.name)
    if a.prepare:
        out=R/('PREFLIGHT-'+a.name+'-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'.json');old.providers(build,env,out);print('No-model provider preflight passed: '+str(out));return 0
    prerequisites(spec)
    assert os.geteuid()==0,'existing supervisor requires root'
    for u in ['llm-deepseek41-api.service','llm-openai-api.service']:assert subprocess.run(['systemctl','is-active','--quiet',u]).returncode!=0,'API still active'
    for u in ['xeon-watch.service','xeon-watch-notify.service']:assert subprocess.run(['systemctl','is-active','--quiet',u]).returncode==0,'monitor inactive'
    stops=[ENGINE/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED']+list((R/'runs').glob('*/STOP_REQUESTED'))
    assert not any(p.exists() for p in stops),'protection stop must be investigated'
    watch=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());assert 0<=time.time()-watch['unix']<=45 and not watch['errors'] and not watch['active_alerts']
    mem={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    assert mem['MemTotal']>=480*1024**2 and mem['MemAvailable']>=360*1024**2 and mem['Mlocked']<=32*1024**2
    gpu=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=list(csv.reader(gpu.strip().splitlines()));assert len(rows)==1 and rows[0][1].strip()==GPU and rows[0][2].strip().upper()=='00000000:84:00.0'
    state_path=R.parent/'STATE.json';state=json.loads(state_path.read_text());assert a.name not in state['runs'],'do not overwrite/retry an existing stage without explicit revision'
    run=R/'runs'/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'Z-'+a.name+'-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True);st=LAB.stat();os.chown(run,st.st_uid,st.st_gid)
    old.providers(build,env,run/'PROVIDER_EXPECTED.json');write(run/'RUN_SPEC.json',spec)
    command=[*base['prefix'],str(R/'driver'),'--study-output',str(run),'--study-cases',str(R/'FROZEN_PLAN.json'),'--run-spec',str(run/'RUN_SPEC.json'),*args]
    # Same JSON encoding as old guard config so public reconstruction stays exact.
    (run/'server-config.json').write_text(json.dumps(dict(command=command,environment=env),indent=2)+'\n');write(run/'BUILD.json',build)
    write(run/'LAUNCH_MANIFEST.json',dict(command=command,environment=env,spec=spec,plan_sha256=sha(R/'FROZEN_PLAN.json'),bank_plan_sha256=sha(R/'capacity.plan'),
        run_spec_sha256=sha(run/'RUN_SPEC.json'),runner_sha256=sha(Path(__file__)),build_sha256=sha(R/'BUILD.json'),GPU_identity_raw=gpu,
        baseline_config_sha256=base['baseline_config_sha256'],model_manifest_sha256=base['model_manifest_sha256'],guard_source_sha256=base['guard_source_sha256'],reserve_GiB=64,budget_seconds=7200))
    state['runs'][a.name]={'path':str(run),'status':'running'};state['phase']='Running '+a.name;write(state_path,state)
    resource.setrlimit(resource.RLIMIT_MEMLOCK,(resource.RLIM_INFINITY,resource.RLIM_INFINITY));sys.path.insert(0,str(ENGINE));guard=module('validation_guard',ENGINE/'supervise.py');guard.ROOT=run;original=guard.check_host;deadline=time.monotonic()+7200
    def strict(initial):
        cpu,available=original(initial)
        if available<64*1024**2:raise RuntimeError('validation RAM reserve below64GiB')
        if time.monotonic()>deadline:raise TimeoutError('validation process2hour limit')
        return cpu,available
    guard.check_host=strict;print('Validation run: '+str(run),flush=True);code=guard.main()
    d=json.loads((run/'RUN.json').read_text()) if (run/'RUN.json').exists() else {};code=code or (0 if d.get('passed') else 1)
    write(run/'RESULTS_SHA256.json',{p.name:sha(p) for p in sorted(run.iterdir()) if p.is_file() and p.name!='RESULTS_SHA256.json'})
    state=json.loads(state_path.read_text());state['runs'][a.name].update(status='passed' if code==0 else 'failed',returncode=code,manifest_sha256=sha(run/'RESULTS_SHA256.json'));write(state_path,state)
    print(json.dumps(dict(run=str(run),name=a.name,code=code,passed=d.get('passed',False))),flush=True);return code
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:raise SystemExit(type(e).__name__+': '+str(e))
