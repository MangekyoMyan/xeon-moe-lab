"""Verify the restored ordinary API without repeating an unchanged long benchmark."""
from pathlib import Path
import fcntl,hashlib,json,subprocess,time,urllib.request
from heldout_probe import ROOT as R, CONFIG, helpers
def main():
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();assert raw==(R/'profiles/final.json').read_bytes();c=json.loads(raw);pid=helpers.api_pid()
    args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    i=next(i for i,x in enumerate(c['command']) if Path(x).name=='llama-server');assert args==c['command'][i:]
    assert args[args.index('-c')+1]=='131072'
    env=dict(x.split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x)
    assert all(env.get(k)==v for k,v in c['environment'].items())
    assert 'GGML_XEON_HOT_FFN' not in env
    maps={str(Path(x.split()[-1]).resolve()) for x in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libggml' in x or 'libllama' in x}
    for entry in c['study_libraries']:
        assert str(Path(entry['file']).resolve()) in maps and hashlib.sha256(Path(entry['file']).read_bytes()).hexdigest()==entry['sha256']
    for entry in c['study_files']:assert hashlib.sha256(Path(entry['file']).read_bytes()).hexdigest()==entry['sha256']
    headers={'Authorization':'Bearer '+Path(c['api_key_file']).read_text().strip(),'Content-Type':'application/json'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f"http://{c['host']}:{c['port']}"
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body).encode()),timeout=120) as response:return json.load(response)
    assert query('/health')['status']=='ok' and not any(x.get('is_processing') for x in query('/slots'))
    response=query('/v1/chat/completions',{'model':c['model_alias'],'messages':[{'role':'user','content':'Reply with only one digit: what is 2 + 3?'}],'temperature':0,'max_tokens':8,'stream':False})
    answer=response['choices'][0]['message']['content'].strip();assert answer=='5'
    watch=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text());assert time.time()-watch['unix']<45 and not watch['errors'] and not watch['active_alerts']
    assert watch['gpu']['ecc_mode']=='Enabled' and abs(watch['gpu']['power_limit_W']-150)<.5
    assert watch['bmc']['fan_mode']=='00' and max(watch['bmc']['fan_percent'])<=70
    services={n:subprocess.check_output(['systemctl','is-active',n],text=True).strip() for n in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']}
    assert all(x=='active' for x in services.values()) and helpers.api_pid()==pid and CONFIG.read_bytes()==raw
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','Mlocked:'))}
    result={'passed':True,'captured_unix':time.time(),'profile':'final','profile_sha256':hashlib.sha256(raw).hexdigest(),'pid':pid,'API_url':url+'/v1','model_alias':c['model_alias'],'context_capacity':131072,
        'command_matches_config':True,'environment_matches_config':True,'hot_backend_enabled':False,'loaded_libraries_verified':c['study_libraries'],'plans_verified':c['study_files'],
        'OpenAI_chat_smoke':{'question':'2+3','answer':answer,'usage':response.get('usage')},'services':services,'GPU':watch['gpu'],'BMC_fan_mode':watch['bmc']['fan_mode'],'BMC_fan_percent':watch['bmc']['fan_percent'],'memory_KiB':memory,
        'feature_scope':'Restored identical immutable16:00checkpoint libraries/config. Earlier image/16k validation preserved in checkpoint-1600; not rerun after this restoration. Separate hot14 image/16k tests passed before rollback.',
        'experimental_profile':'hot14-reviewed','decision_record':str(R/'EVENING_RESULTS_JA.txt')}
    (R/'EVENING_FINAL_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'profile':'final','API_url':result['API_url'],'answer':answer,'services':services}),flush=True)
if __name__=='__main__':main()
