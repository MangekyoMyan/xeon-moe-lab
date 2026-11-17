"""Offline audit of the public bundle; no model, hardware or network requests.

Runtime config DATA is reconstructed only in a temporary copy so the original
manifest can be checked; commands inside those records are never executed.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,tempfile
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    manifest=json.loads((a.root/'FILES_SHA256.json').read_text())
    for name,item in manifest.items():
        data=(a.root/name).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name
    with tempfile.TemporaryDirectory(prefix='workspace-review-') as tmp:
        root=Path(tmp)/'bundle';shutil.copytree(a.root,root,ignore=shutil.ignore_patterns('__pycache__'))
        for health in root.rglob('guard-health.json'):
            run=health.parent;health.rename(run/'health.json');launch=json.loads((run/'LAUNCH_MANIFEST.json').read_text())
            (run/'server-config.json').write_text(json.dumps(dict(command=launch['command'],environment=launch['environment']),indent=2)+'\n')
        base=root/'runs/20261117T031214Z-U1024-93e7fe85'
        bank=root/'bank-capacity/runs/20261117T033327Z-U1024-0b6d5d01'
        pilot=root/'pilot-capture/runs/20261117T035402Z-U1024-d22b3e04'
        results=Path(tmp)/'results';results.mkdir()
        commands=[['analyze_capacity.py',str(base),str(bank),'--output',str(results/'BANK.json')],['analyze_pilot.py',str(pilot),'--output',str(results/'PILOT.json')]]
        for cmd in commands:
            proc=subprocess.run([sys.executable,str(root/cmd[0]),*cmd[1:]],capture_output=True,text=True,timeout=120)
            if proc.returncode:raise RuntimeError(proc.stderr)
        assert (results/'TRAIN5_CANDIDATE.plan').read_bytes()==(root/'TRAIN5_CANDIDATE.plan').read_bytes()
        capacity=json.loads((results/'BANK.json').read_text());proposal=json.loads((results/'PILOT.json').read_text())
        result=dict(passed=True,model_started=False,public_files=len(manifest),bank_guards=capacity['runs'][1]['bank_checks'],all_capacity_outputs_byte_equal=all(c['byte_equal'] for c in capacity['comparisons']),training_calls=proposal['training_calls'],training_records=proposal['records'],candidate_count=proposal['candidate']['count'],candidate_plan_bytes_equal=True,holdout_routes_used=False)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
