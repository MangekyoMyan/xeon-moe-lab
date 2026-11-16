"""One final functional/state snapshot after the selected profile and feature smoke tests."""
from pathlib import Path
import fcntl,hashlib,json,platform,subprocess,time,urllib.request
from heldout_probe import ROOT,CONFIG,helpers,private_write
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    lock=(ROOT/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();profile=json.loads(raw);assert raw==(ROOT/'profiles/final.json').read_bytes()
    key=Path(profile['api_key_file']).read_text().strip();headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f"http://{profile['host']}:{profile['port']}"
    def query(route,body=None):
        with opener.open(urllib.request.Request(url+route,headers=headers,data=None if body is None else json.dumps(body).encode()),timeout=60) as response:return json.load(response)
    assert query('/health')['status']=='ok' and not any(s.get('is_processing') for s in query('/slots'))
    before=helpers.watch();pid=helpers.api_pid();started=time.monotonic()
    stat_before=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();start_ticks=stat_before[19]
    actual_command=[x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    command=profile['command'];server_index=next(i for i,x in enumerate(command) if x.endswith('/llama-server'))
    assert actual_command==command[server_index:]
    actual_context=int(actual_command[actual_command.index('-c')+1]);assert actual_context==profile['context']==131072
    reply=query('/v1/chat/completions',{'model':profile['model_alias'],'messages':[{'role':'system','content':'Answer with a single decimal integer and no other text.'},{'role':'user','content':'What is 2 + 3?'}],'temperature':0,'seed':42,'max_tokens':8,'stream':False})
    chat_wall=time.monotonic()-started
    answer=reply['choices'][0]['message']['content'].strip();assert answer=='5',answer
    after=helpers.watch();assert before['host_counters']==after['host_counters']
    assert after['gpu']['power_limit_W']==150 and after['gpu']['ecc_mode']=='Enabled'
    assert helpers.api_pid()==pid and CONFIG.read_bytes()==raw
    assert query('/health')['status']=='ok' and not any(s.get('is_processing') for s in query('/slots'))
    mappings=sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'libggml' in line or 'libllama' in line})
    for item in profile['study_libraries']:
        assert item['file'] in mappings and digest(Path(item['file']))==item['sha256']
    for item in profile['study_files']:assert digest(Path(item['file']))==item['sha256']
    vision=[(p,json.loads(p.read_text())) for p in (ROOT/'vision').glob('*-final-newspaper/RESULT.json')]
    long=[(p,json.loads(p.read_text())) for p in (ROOT/'long-cache').glob('*-final-16k/RESULT.json')]
    assert len(vision)==len(long)==1
    assert vision[0][1]['pid']==pid and vision[0][1]['error'] is None and vision[0][1]['headline_terms_present']
    assert vision[0][1]['config_sha256']==hashlib.sha256(raw).hexdigest()
    assert long[0][1]['pid']==pid and long[0][1]['passed'] and long[0][1]['error'] is None
    assert long[0][1]['config_sha256']==hashlib.sha256(raw).hexdigest()
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key0,rest=line.split(':',1)
        if key0 in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:memory[key0+'_KiB']=int(rest.split()[0])
    board={}
    for name in ['board_vendor','board_name','board_version','bios_version']:
        path=Path('/sys/class/dmi/id')/name
        if path.exists():board[name]=path.read_text().strip()
    model_names=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')})
    services={unit:subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip() for unit in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']}
    assert services['llm-deepseek41-api.service']=='active' and services['xeon-watch.service']=='active'
    assert helpers.api_pid()==pid and CONFIG.read_bytes()==raw and Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==start_ticks
    record={'passed':True,'unix':time.time(),'pid':pid,'pid_start_ticks':start_ticks,'actual_command':actual_command,'profile_sha256':hashlib.sha256(raw).hexdigest(),'API_url':url+'/v1','model_alias':profile['model_alias'],'context_capacity':actual_context,
        'chat_smoke':{'answer':answer,'usage':reply.get('usage'),'wall_seconds':chat_wall},'vision_result':str(vision[0][0]),'long_cache_result':str(long[0][0]),
        'libraries':mappings,'verified_library_hashes':profile['study_libraries'],'verified_files':profile['study_files'],'services':services,'watch':after,'memory':memory,'board':board,'cpu_models':model_names,'uname':list(platform.uname()),
        'operating_policy':'GPU150W,ECC enabled, fan limit70 unchanged; no model files modified; boot startup configuration not changed by this study','rollback_profile':'baseline'}
    target=ROOT/'FINAL_STATUS.json'
    if target.exists():raise RuntimeError('Final snapshot already exists; inspect before replacing')
    private_write(target,record);print(json.dumps({k:record[k] for k in ['passed','pid','API_url','model_alias','context_capacity','services']},indent=2))
if __name__=='__main__':main()
