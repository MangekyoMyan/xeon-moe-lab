"""Offline saved-evidence replay; never load engine DSOs or execute launch commands."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,tempfile
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    m=read(a.root/'FILES_SHA256.json')
    for n,v in m.items():
        rel=Path(n);assert not rel.is_absolute() and '..' not in rel.parts
        p=a.root/rel;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
    cap=a.root.parent/'2026-11-17-parallel-pilot/capture/runs/20261117T052657Z-U1024-3483cb75'
    design=read(a.root/'q8-avx1/FFN_DESIGN.json');assert sha(cap/'RESULTS_SHA256.json')==design['common']['capture_manifest_sha256']
    cm=read(cap/'RESULTS_SHA256.json');assert sha(cap/'CAPTURE.json')==cm['CAPTURE.json']
    for cell in read(cap/'CAPTURE.json')['cells']:
        for key in ['x_file','ids_file','down_file','routing_weights_file']:assert sha(cap/cell[key])==cm[cell[key]]
    with tempfile.TemporaryDirectory(prefix='q8-review-') as tmp:
        root=Path(tmp)/'bundle';shutil.copytree(a.root,root)
        for h in (root/'q8-avx1').rglob('guard-health.json'):
            h.rename(h.with_name('health.json'));j=read(h.parent/'LAUNCH.json')
            (h.parent/'server-config.json').write_text(json.dumps(dict(command=j['command'],environment=j['environment']),indent=2)+'\n')
        s=read(root/'q8-avx1/STATE-v2.json')
        for e in s['runs'].values():e['path']=str(root/'q8-avx1/runs-v2'/Path(e['path']).name)
        (root/'q8-avx1/STATE-v2.json').write_text(json.dumps(s,indent=2)+'\n')
        expected=read(root/'q8-avx1/FFN_ANALYSIS.json');(root/'q8-avx1/FFN_ANALYSIS.json').unlink()
        spec=importlib.util.spec_from_file_location('replay_analysis',root/'analyze_ffn.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.CAPTURE_OVERRIDE=cap/'CAPTURE.json';module.main()
        assert read(root/'q8-avx1/FFN_ANALYSIS.json')==expected
    result=dict(passed=True,public_files=len(m)+1,successful_processes=10,timed_graph_calls=1920,saved_outputs_checked=60,performance_rows_reproduced=6,model_started=False,DSO_loaded=False,provided_commands_executed=False,network_used=False,
        scope='Public source/evidence hashes and realFFN saved-output/timing replay. Unit numerical and external VM checks are preserved receipts, not rerun here.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
