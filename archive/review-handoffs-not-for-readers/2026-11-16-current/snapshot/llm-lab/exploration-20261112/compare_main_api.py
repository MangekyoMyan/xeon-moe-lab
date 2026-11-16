from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parent
def run(script,*args):subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    result={'runs':[],'complete':False,'error':None};output=R/'cpu-coordination-mx/main-last/API_RUNS.json'
    if output.exists():raise RuntimeError('CPU placement comparison already exists')
    try:
        for profile,labels in [('main-first-control',['main-control-r1']),('main-last',['main-candidate-r1','main-candidate-r2']),('main-first-control',['main-control-r2'])]:
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                for workload in ['count','code']:
                    run('bench_api.py','--label',label,'--workload',workload,'--context','128','--tokens','96')
                    matches=list((R/'runs').glob('*-'+label+'-'+workload+'-plain-c128-n96/RESULT.json'));assert len(matches)==1
                    result['runs'].append({'profile':profile,'label':label,'workload':workload,'result':str(matches[0])});output.write_text(json.dumps(result,indent=2)+'\n')
        result['complete']=True
    except BaseException as error:result['error']=repr(error);raise
    finally:output.write_text(json.dumps(result,indent=2)+'\n')
    print('Same-binary CPU placement ABBA completed; no automatic adoption.',flush=True)
if __name__=='__main__':main()
