"""Exercise the active-API refusal with mocked service replies, never a model launch."""
from pathlib import Path
import importlib.util,json,subprocess,sys
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capture_under_test',R/'run_capture.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
events=[];original_run=m.subprocess.run;original_popen=m.subprocess.Popen
def fake_run(argv,**kw):
    events.append(argv)
    assert argv==['systemctl','is-active','--quiet','llm-deepseek41-api.service'],argv
    return subprocess.CompletedProcess(argv,0)
def forbidden(*args,**kw):raise AssertionError('Popen must not be reached')
try:
    m.subprocess.run=fake_run;m.subprocess.Popen=forbidden;sys.argv=['run_capture.py']
    try:m.main();raise AssertionError('Active API accepted')
    except RuntimeError as e:assert 'is running; not stopping it automatically' in str(e);message=str(e)
finally:m.subprocess.run=original_run;m.subprocess.Popen=original_popen
(R/'LAUNCH_REFUSAL_TEST.json').write_text(json.dumps({'passed':True,'scope':'Mocked systemctl active reply; preparation reads real pinned files, no real service/GPU/model operation','mock_calls':events,'result':message,'model_process_created':False},indent=2)+'\n');print('Active-API branch refused without reaching process creation')
