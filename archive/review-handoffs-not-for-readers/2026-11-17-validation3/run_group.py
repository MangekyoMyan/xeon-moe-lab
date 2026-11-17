"""Explicit stage controller. Stops on any failure; never changes or starts the API."""
from pathlib import Path
import argparse,fcntl,json,subprocess,sys
from verify_evidence import load_run
R=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['quality','timing','answer']);a=ap.parse_args()
    lock=(R/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=[s for s in json.loads((R/'caller/DESIGN.json').read_text())['stages'] if s['stage']==a.stage]
    for spec in specs:
        state=json.loads((R/'STATE.json').read_text());name=spec['name']
        if name in state['runs']:
            assert state['runs'][name]['status']=='passed','existing running/failed stage must be investigated'
            load_run(name);print('Verified existing completed stage '+name,flush=True);continue
        print('Starting '+name,flush=True)
        subprocess.run([sys.executable,str(R/'caller/run_validation.py'),name],check=True,timeout=7500)
        load_run(name);print('Completed '+name,flush=True)
    if a.stage in ['quality','timing']:
        p=R/('QUALITY_GATE.json' if a.stage=='quality' else 'TIMING_COMPLETE.json')
        if not p.exists():subprocess.run([sys.executable,str(R/'verify_evidence.py'),a.stage],check=True,timeout=180)
    print('Group complete: '+a.stage,flush=True)
if __name__=='__main__':main()
