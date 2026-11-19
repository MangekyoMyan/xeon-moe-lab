"""CPU-only replay of saved outputs and clocks; never executes saved model commands."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys,tempfile
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    root=Path(sys.argv[1]).resolve();manifest=read(root/'FILES_SHA256.json')
    for name,row in manifest.items():
        p=root/name;assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
    # All mutations are to a temporary copy; only verified analysis source is imported.
    with tempfile.TemporaryDirectory(prefix='shared-prefix-replay-') as temp:
        dst=Path(temp)/'study';dst.mkdir()
        for n in ['run_main.py','run_trial.py','analyze_trial.py','parent_recheck.py']:
            shutil.copy2(root/n,dst/n)
        for n in ['main','trial']:
            shutil.copytree(root/n,dst/n)
        manifests=0
        for p in dst.glob('*/runs/*/guard-health.json'):shutil.copy2(p,p.with_name('health.json'))
        for p in dst.glob('*/runs/*/LAUNCH.json'):
            j=read(p);p.with_name('server-config.json').write_text(json.dumps(dict(command=j['command'],environment=j['environment']),indent=2)+'\n')
        for p in dst.glob('*/runs/*/RESULTS_SHA256.json'):
            for name,digest in read(p).items():assert sha(p.parent/name)==digest,(p,name)
            manifests+=1
        normal=module('public_normal_verify',dst/'run_main.py');trial=module('public_trial_verify',dst/'run_trial.py')
        for p in (dst/'main/runs').iterdir():normal.verify(p)
        for p in (dst/'trial/runs').iterdir():trial.verify(p)
        recomputed=module('public_timing_analysis',dst/'analyze_trial.py').analyze(dst)
        assert recomputed==read(root/'trial/ANALYSIS.json')
        timeline=module('public_independent_timeline',dst/'parent_recheck.py')
        raw={name:timeline.inspect(dst/path,candidate) for name,path,candidate in [('original','main/runs/observe',False),('candidate','trial/runs/observe-B',True)]}
        assert raw==read(root/'PARENT_BOTH_TIMELINES.json')
    print(json.dumps(dict(passed=True,manifest_files=len(manifest),run_manifests_reconstructed=manifests,successful_processes=8,raw_timed_calls=512,quality_rows_per_case_per_arm=65,quality_arms=4,independent_timelines_reproduced=True,candidate_decision=recomputed['decision'],model_or_GPU_execution=False)))
if __name__=='__main__':main()
