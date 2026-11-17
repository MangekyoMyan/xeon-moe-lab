"""One predeclared case-only ABBA; initial six responses must already be preserved."""
from pathlib import Path
import fcntl,json,subprocess,sys
from verify_evidence import load_run,common,sha,write
R=Path(__file__).resolve().parent
def main():
    parent=json.loads((R.parent/'STATE.json').read_text())
    for n in ['answer-0-F','answer-1-P','answer-2-F','answer-3-P']:assert parent['runs'][n]['status']=='passed'
    f=(R.parent/'.controller.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=json.loads((R/'caller/DESIGN.json').read_text())['stages'];runs=[]
    for s in specs:
        state=json.loads((R/'STATE.json').read_text())
        if s['name'] not in state['runs']:
            print('Starting supplemental '+s['name'],flush=True)
            subprocess.run([sys.executable,str(R/'caller/run_validation.py'),s['name']],check=True,timeout=7500)
        runs.append(load_run(s['name']));print('Verified supplemental '+s['name'],flush=True)
    common(runs);assert sum(len(x[1]['blocks'])*64 for x in runs)==256
    write(R/'COMPLETE.json',dict(passed=True,timed_calls=256,runs=[dict(path=str(x[0]),manifest_sha256=sha(x[0]/'RESULTS_SHA256.json')) for x in runs]));print('Supplemental ABBA complete',flush=True)
if __name__=='__main__':main()
