"""Optional threshold32 experiment after the final CPU setting is selected."""
from pathlib import Path
import argparse,copy,datetime,json,subprocess,time
from zoneinfo import ZoneInfo
from heldout_probe import CONFIG,helpers
R=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-profile',required=True);a=p.parse_args();assert a.base_profile.replace('-','').isalnum()
    base=json.loads((R/'profiles'/(a.base_profile+'.json')).read_text());candidate=copy.deepcopy(base);env=candidate['environment']
    zone=ZoneInfo('Asia/Tokyo');latest=datetime.datetime(2026,11,12,15,25,tzinfo=zone);deadline=datetime.datetime(2026,11,12,15,45,tzinfo=zone)
    if datetime.datetime.now(zone)>latest:raise RuntimeError('Too late to start optional threshold32 experiment')
    def run(script,*args):
        remaining=deadline.timestamp()-time.time()
        if remaining<=0:raise TimeoutError('Optional experiment deadline reached')
        subprocess.run(['python3',str(R/script),*args],check=True,timeout=remaining)
    for key in ['GGML_OP_OFFLOAD_MIN_BATCH','GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH']:
        assert env[key]=='64';env[key]='32'
    name='short-gpu32';candidate['optimization_study'].update(profile=name,status='experimental-short-input-only',offload_threshold=32)
    candidate['gpu_stream_profile'].update(name='gpu3-stream32-experimental',full_model_validation='pending short-input comparison',decode_policy='CPU experts except resident layers; GPU offload batch>=32 for IQ and MXFP4')
    output=R/'profiles'/(name+'.json')
    if output.exists():raise RuntimeError('Threshold32 profile already exists; inspect before repeating')
    output.write_text(json.dumps(candidate,indent=2)+'\n');output.chmod(0o600)
    record={'base_profile':a.base_profile,'started_unix':time.time(),'complete':False,'error':None,'recovery':None};status=R/'short-prefill/CONTROLLER.json';status.parent.mkdir(exist_ok=True)
    try:
        for profile,labels in [(a.base_profile,['threshold64-r1']),(name,['threshold32-r1','threshold32-r2']),(a.base_profile,['threshold64-r2'])]:
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                run('short_prefill_probe.py','--label',label)
        record['complete']=True
    except BaseException as error:
        record['error']=repr(error)
        try:
            if (CONFIG.parent/'STOP_REQUESTED').exists():raise RuntimeError('Protection stop present; restoration intentionally not attempted')
            helpers.watch()
            if CONFIG.read_bytes()==(R/'profiles'/(a.base_profile+'.json')).read_bytes():record['recovery']='Base configuration already selected; no forced restart'
            else:
                # The switch checks idle slots and service state before stopping anything.
                subprocess.run(['python3',str(R/'switch_and_wait.py'),a.base_profile],check=True,timeout=900)
                record['recovery']='Restored base profile after safety and idle checks'
        except BaseException as recovery_error:record['recovery']='Not restored: '+repr(recovery_error)
        raise
    finally:record['finished_unix']=time.time();status.write_text(json.dumps(record,indent=2)+'\n')
    print('Threshold32 boundary comparison completed; no automatic adoption.',flush=True)
if __name__=='__main__':main()
