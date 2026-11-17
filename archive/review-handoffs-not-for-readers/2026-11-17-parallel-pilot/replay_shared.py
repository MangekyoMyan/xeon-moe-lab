"""Offline verification of the shared evidence; never execute stored launch commands."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,tempfile

def read(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    m=read(a.root/'FILES_SHA256.json')
    for n,v in m.items():
        b=(a.root/n).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'],n
    with tempfile.TemporaryDirectory(prefix='parallel-review-') as tmp:
        root=Path(tmp)/'bundle';shutil.copytree(a.root,root,ignore=shutil.ignore_patterns('__pycache__'))
        for h in root.rglob('guard-health.json'):
            run=h.parent;h.rename(run/'health.json');j=read(run/'LAUNCH_MANIFEST.json')
            (run/'server-config.json').write_text(json.dumps(dict(command=j['command'],environment=j['environment']),indent=2)+'\n')
        state=read(root/'STATE.json');res=Path(tmp)/'results';res.mkdir()
        cap=root/'capture/runs/20261117T052657Z-U1024-3483cb75';op=root/'replay/runs/20261117T053052Z-U1024-1563d936'
        old=a.root.parent/'2026-11-17-gpu-workspace/pilot-capture/runs/20261117T035402Z-U1024-d22b3e04/text_cache_budget_ja.json'
        assert old.is_file(),'Keep previous published workspace snapshot beside this bundle'
        cmds=[['analyze.py','capture',str(cap),'--old-ja',str(old),'--output',str(res/'capture.json')],
              ['analyze.py','replay',str(op),'--output',str(res/'operator.json')]]
        if state.get('integrated_all_completed'):
            cmds.append(['analyze_integrated.py',*[str(root/state[k]) for k in ['integrated_F_before','integrated_CSP','integrated_F_after']], '--output',str(res/'integrated.json')])
        for cmd in cmds:
            p=subprocess.run([sys.executable,str(root/cmd[0]),*cmd[1:]],capture_output=True,text=True,timeout=300)
            if p.returncode:raise RuntimeError(p.stderr)
        result=dict(passed=True,public_files=len(m),model_started=False,network_used=False,stored_commands_executed=False,
            capture_all333_routes_equal=read(res/'capture.json')['all_first9_333_routes_equal'],operator_cells=len(read(res/'operator.json')['cells']))
        if (res/'integrated.json').exists():
            j=read(res/'integrated.json');original=read(root/'INTEGRATED_ANALYSIS.json');assert j['timing_summary']==original['timing_summary'] and j['quality_comparisons']==original['quality_comparisons']
            result['integrated_reproduced']=True;result['integrated_speed_vs_F_bracket']=j['speed_vs_F_bracket']
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
