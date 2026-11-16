"""Code-identical native pool poll0/50 ABBA; preserve the 16:00 checkpoint."""
from pathlib import Path
import hashlib,json,subprocess,time
R=Path(__file__).resolve().parent
def run(script,*args):
    subprocess.run(['python3',str(R/script),*args],check=True)
def main():
    folder=R/'native-poll';folder.mkdir(exist_ok=True)
    state=folder/'CONTROLLER.json'
    if state.exists():raise RuntimeError('Existing experiment must be inspected before repeat')
    baseline=(R/'profiles/native-pool.json').read_bytes()
    c=json.loads(baseline)
    for flag in ['--poll','--poll-batch']:
        i=c['command'].index(flag);assert c['command'][i+1]=='0';c['command'][i+1]='50'
    c['optimization_study']['profile']='native-poll50'
    c['optimization_study']['status']='experimental-code-identical-poll-change'
    target=R/'profiles/native-poll50.json'
    if target.exists():raise RuntimeError('Unexpected preexisting candidate profile')
    target.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');target.chmod(0o600)
    record={'scope':'Native graph executor code identical; poll0 vs50 for both decode/prefill. Existing frozen inputs, ABBA,128complete generated IDs. No other code/config changes.',
            'started_unix':time.time(),'complete':False,'error':None,'completed_labels':[],
            'profiles':{'native-pool':hashlib.sha256(baseline).hexdigest(),'native-poll50':hashlib.sha256(target.read_bytes()).hexdigest()}}
    def save():state.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        for profile,labels in [('native-pool',['poll0-r1']),('native-poll50',['poll50-r1','poll50-r2']),('native-pool',['poll0-r2'])]:
            assert hashlib.sha256((R/'profiles'/f'{profile}.json').read_bytes()).hexdigest()==record['profiles'][profile]
            run('switch_and_wait.py',profile)
            for label in labels:
                run('bench_api.py','--label',label+'-warm','--context','1024','--tokens','8')
                run('fresh_probe.py','--label',label)
                record['completed_labels'].append(label);save()
        run('analyze_heldout.py','--control','poll0-r1','poll0-r2','--candidate','poll50-r1','poll50-r2','--out','native-poll/LIVE_ABBA.json','--scope',record['scope'],'--final')
        run('switch_and_wait.py','final')
        record['complete']=True
    except BaseException as e:record['error']=repr(e);raise
    finally:record['finished_unix']=time.time();save()
if __name__=='__main__':main()
