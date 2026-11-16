"""Confirm a promising wait-policy screen result without perf instrumentation."""
from pathlib import Path
import argparse,hashlib,json,subprocess
R=Path(__file__).resolve().parent
def run(script,*args):subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args()
    manifest=json.loads((R/'cpu-wait/PROFILES.json').read_text())
    expected={x['name']:x['sha256'] for x in manifest['profiles']}
    assert a.candidate in expected and a.candidate!='wait-t12-default'
    for profile in [a.candidate,'wait-t12-default']:
        assert hashlib.sha256((R/'profiles'/(profile+'.json')).read_bytes()).hexdigest()==expected[profile]
    record={'candidate':a.candidate,'runs':[],'complete':False,'error':None};output=R/'cpu-wait/PLAIN_CONFIRMATION.json'
    if output.exists():raise RuntimeError('A plain confirmation already exists; inspect it before repeating')
    try:
        for profile,labels in [('wait-t12-default',['waitplain-control-r1']),(a.candidate,['waitplain-candidate-r1','waitplain-candidate-r2']),('wait-t12-default',['waitplain-control-r2'])]:
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                for workload in ['count','code']:
                    run('bench_api.py','--label',label,'--workload',workload,'--context','128','--tokens','128')
                    matches=list((R/'runs').glob('*-'+label+'-'+workload+'-plain-c128-n128/RESULT.json'));assert len(matches)==1
                    record['runs'].append({'profile':profile,'label':label,'workload':workload,'result':str(matches[0])});output.write_text(json.dumps(record,indent=2)+'\n')
        record['complete']=True
    except BaseException as error:record['error']=repr(error);raise
    finally:output.write_text(json.dumps(record,indent=2)+'\n')
    print('Uninstrumented wait-policy ABBA completed; no automatic adoption.',flush=True)
if __name__=='__main__':main()
