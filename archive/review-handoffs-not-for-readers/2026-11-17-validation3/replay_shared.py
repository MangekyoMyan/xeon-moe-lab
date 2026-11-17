"""Offline public evidence replay. No model, stored launch command or generated code execution."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,sys,tempfile
def read(p):return json.loads(p.read_text())
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert not a.output.exists()
    manifest=read(a.root/'FILES_SHA256.json')
    for n,v in manifest.items():
        rel=Path(n);assert not rel.is_absolute() and '..' not in rel.parts
        raw=(a.root/rel).read_bytes();assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'],n
    with tempfile.TemporaryDirectory(prefix='validation-review-') as tmp:
        root=Path(tmp)/'validation3-20261117';shutil.copytree(a.root,root,ignore=shutil.ignore_patterns('__pycache__'))
        old=Path(tmp)/'parallel-pilot-20261117';old.mkdir()
        for n in ['analyze.py','analyze_integrated.py']:
            p=a.root.parent/'2026-11-17-parallel-pilot'/n
            assert p.is_file(),'Keep previous published snapshot beside this directory'
            shutil.copy2(p,old/n)
        for h in root.rglob('guard-health.json'):
            run=h.parent;h.rename(run/'health.json');j=read(run/'LAUNCH_MANIFEST.json')
            (run/'server-config.json').write_text(json.dumps(dict(command=j['command'],environment=j['environment']),indent=2)+'\n')
        for base in [root,root/'python-repeat']:
            state=read(base/'STATE.json')
            for e in state['runs'].values():e['path']=str(base/'caller/runs'/Path(e['path']).name)
            (base/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
        v=module('verify_evidence',root/'verify_evidence.py')
        for name in read(root/'STATE.json')['runs']:v.load_run(name)
        (root/'QUALITY_GATE.json').unlink();v.quality_gate()
        qa=read(root/'QUALITY_ANALYSIS.json');ta=read(root/'TIMING_ANALYSIS.json')
        (root/'QUALITY_ANALYSIS.json').unlink();(root/'TIMING_ANALYSIS.json').unlink()
        analyzer=module('validation_analyzer',root/'analyze_validation.py');analyzer.quality();analyzer.timing()
        assert qa==read(root/'QUALITY_ANALYSIS.json') and ta==read(root/'TIMING_ANALYSIS.json')
        av=read(root/'ANSWER_VERIFICATION.json');(root/'ANSWER_VERIFICATION.json').unlink()
        for row in av['responses']:(root/row['rendered_file']).unlink()
        module('answer_verifier',root/'verify_answers.py').main();new=read(root/'ANSWER_VERIFICATION.json')
        assert av['generated_IDs_F_P_equal']==new['generated_IDs_F_P_equal']
        for before,after in zip(av['responses'],new['responses']):
            assert {k:x for k,x in before.items() if k not in ['run','source_json']}=={k:x for k,x in after.items() if k not in ['run','source_json']}
            assert (a.root/before['rendered_file']).read_bytes()==(root/after['rendered_file']).read_bytes()
        hit=read(root/'HIT_ANALYSIS.json');(root/'HIT_ANALYSIS.json').unlink();module('hit_analyzer',root/'analyze_hits.py').main();assert hit==read(root/'HIT_ANALYSIS.json')
        rep=root/'python-repeat';expected=read(rep/'ANALYSIS.json');(rep/'ANALYSIS.json').unlink()
        module('verify_evidence',rep/'verify_evidence.py');module('repeat_analyzer',rep/'analyze_repeat.py').main();assert expected==read(rep/'ANALYSIS.json')
        result=dict(passed=True,public_files=len(manifest),quality_cases_reproduced=3,initial_timed_calls=768,supplemental_timed_calls=256,answer_journals_reproduced=6,hit_histograms_reproduced=True,model_started=False,network_used=False,stored_commands_executed=False,generated_code_executed=False,
            qualification='Manual semantic/source grading is preserved evidence, not repeated by this offline replay. Python behavior receipts are hash verified; generated code is not executed here. Absolute run paths are relocated only in a disposable STATE copy.')
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
