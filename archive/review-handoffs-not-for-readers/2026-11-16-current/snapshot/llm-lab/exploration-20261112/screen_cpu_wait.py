"""Explicit sequential screen; all candidates retain the same arithmetic libraries."""
from pathlib import Path
import argparse,json,subprocess,time
R=Path(__file__).resolve().parent
def run(script,*args):subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-profile',required=True);a=p.parse_args()
    run('prepare_cpu_wait_profiles.py','--base-profile',a.base_profile,'--include-six')
    output=R/'cpu-wait/SCREEN_RUNS.json'
    if output.exists():raise RuntimeError('Screen already exists; inspect it before any new experiment')
    record={'base_profile':a.base_profile,'started_unix':time.time(),'runs':[],'complete':False,'error':None}
    def save():output.write_text(json.dumps(record,indent=2)+'\n')
    save()
    order=[('wait-t12-default',1),('wait-t24-default',1),('wait-t24-spin10000',1),('wait-t24-passive',1),('wait-t12-passive',1),('wait-t12-spin10000',1),('wait-t6-default',1),('wait-t12-default',2)]
    try:
        for profile,iteration in order:
            run('switch_and_wait.py',profile)
            label=f'cpuwait-{profile}-r{iteration}'
            run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
            for workload in ['count','code']:
                run('bench_cpu_wait.py','--label',label,'--workload',workload,'--context','128','--tokens','96')
                matches=list((R/'runs').glob('*-'+label+'-'+workload+'-plain-c128-n96/RESULT.json'))
                assert len(matches)==1
                record['runs'].append({'profile':profile,'iteration':iteration,'workload':workload,'result':str(matches[0])});save()
        record['complete']=True
    except BaseException as error:record['error']=repr(error);raise
    finally:record['finished_unix']=time.time();save()
    print('CPU thread/wait screen complete; no winner adopted automatically.',flush=True)
if __name__=='__main__':main()
