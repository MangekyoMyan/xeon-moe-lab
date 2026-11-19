"""Recheck public saved evidence only; no model/ELF/launch-command execution."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    m=read(a.root/'FILES_SHA256.json')
    for n,v in m.items():
        p=Path(n);assert not p.is_absolute() and '..' not in p.parts
        p=a.root/p;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
    cap=a.root.parent/'2026-11-17-parallel-pilot/capture/runs/20261117T052657Z-U1024-3483cb75'
    pin=read(a.root/'h1/CAPTURE_PIN.json');assert sha(cap/'RESULTS_SHA256.json')==pin['files']['RESULTS_SHA256.json']
    cm=read(cap/'RESULTS_SHA256.json')
    for name in ['CAPTURE.json',*[f'cell{i}-{s}' for i in range(8) for s in ['down.f32','x.f32','ids.i32','weights.f32']]]:assert sha(cap/name)==cm[name]
    with tempfile.TemporaryDirectory(prefix='xeon-three-tracks-') as tmp:
        root=Path(tmp)/'bundle';shutil.copytree(a.root,root)
        for component in ['fbc','h1','cpu08']:
            d=root/component
            for h in (d/'runs').glob('*/guard-health.json'):
                h.rename(h.with_name('health.json'));launch=read(h.parent/'LAUNCH.json')
                (h.parent/'server-config.json').write_text(json.dumps(dict(command=launch['command'],environment=launch['environment']),indent=2)+'\n')
            state=read(d/'STATE.json')
            for x in state['runs'].values():x['path']=str(d/'runs'/Path(x['path']).name)
            (d/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
        fbc=module('verify_fbc',root/'verify_fbc.py');old_gate=(root/'fbc/QUALITY_GATE.json').read_bytes();(root/'fbc/QUALITY_GATE.json').unlink();fbc.gate();new_gate=read(root/'fbc/QUALITY_GATE.json');expected=json.loads(old_gate)
        assert new_gate['case_hashes']==expected['case_hashes'] and new_gate['graphs']==expected['graphs'];(root/'fbc/QUALITY_GATE.json').write_bytes(old_gate)
        h1=module('verify_h1',root/'verify_h1.py');h1.REFERENCE_CAPTURE_ROOT=cap
        v8=module('verify_08',root/'verify_08.py');v8.REFERENCE_CAPTURE_ROOT=cap
        for x in read(root/'cpu08/STATE.json')['runs'].values():v8.verify(Path(x['path']))
        analysis=module('analyze_stages',root/'analyze_stages.py')
        for name,fn in [('fbc',analysis.fbc),('h1',analysis.h1),('cpu08',analysis.cpu08)]:
            p=root/name/'ANALYSIS.json';expected=read(p);p.unlink();fn();assert read(p)==expected
        result=dict(passed=True,public_files=len(m)+1,successful_processes=23,FBC_quality_and_original_DAG_reproduced=True,FBC_timed_calls=768,H1_steady_observations=1536,CPU08_timed_graph_calls=960,all_analysis_rows_reproduced=True,
            model_started=False,DSO_loaded=False,stored_commands_executed=False,network_used=False,scope='Artifact/data/timing replay. Provider/build/restoration are saved receipts, not new runtime observations. Only external capture paths and temporary state paths are relocated.')
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
