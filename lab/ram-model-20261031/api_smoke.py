#!/usr/bin/env python3
"""Temporary loopback server: small -> large -> small, always stopped afterward."""
import importlib.util,json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;assert os.geteuid()==0
spec=importlib.util.spec_from_file_location('tc_api_helper',R.parent/'prefill-study-20261030/screen.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
result={'started':time.time(),'requests':[],'production_config_changed':False}
p=subprocess.Popen(['python3',str(R/'supervise.py')],start_new_session=True)
try:
    for i in range(150):
        if p.poll() is not None:raise RuntimeError('Temporary supervisor exited before readiness')
        try:
            if helper.json_request('/health').get('status')=='ok':break
        except (OSError,ValueError):pass
        time.sleep(2)
    else:raise RuntimeError('Temporary API readiness timeout')
    pid=json.loads((R/'health.json').read_text())['pid'];result['model_pid']=pid
    long='\n'.join(f'Record {i:03}: module {i%7} handles {i*13} items and stores results in region {i%5}.' for i in range(1,101))+'\nSummarize these records in one sentence.'
    for name,text in [('small-first','こんにちは。簡単な挨拶を一文でお願いします。'),('large-after-small',long),('small-after-large','17と25の合計を、数字だけで答えてください。')]:
        result['requests'].append(helper.query(name,text,pid))
        (R/'API_SMOKE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Additional free generation, separate from the three historical requests.
    original_request=helper.request
    def extended_request(path,payload=None,timeout=900):
        if payload is not None:payload=dict(payload,max_tokens=128)
        return original_request(path,payload,timeout)
    helper.request=extended_request
    result['requests'].append(helper.query('japanese-free-generation',
        'CPUとRAMの役割の違いを初心者に説明してください。日本語で短い三文にまとめてください。',pid))
    helper.request=original_request
    result['success']=True
except BaseException as e:
    result['success']=False;result['error']=repr(e);raise
finally:
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:p.wait(timeout=70)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
    result['supervisor_returncode']=p.returncode;result['finished']=time.time()
    if (R/'health.json').exists():result['health']=json.loads((R/'health.json').read_text())
    (R/'API_SMOKE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('API smoke complete, success',result.get('success'),flush=True)
