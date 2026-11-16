#!/usr/bin/env python3
"""Explicit study profile switch, bounded readiness tracking, persistent progress."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,time,urllib.request,urllib.error
R=Path(__file__).resolve().parent;LAB=R.parent;ENGINE=LAB/'deepseek41-bringup-20261101'
CONFIG=ENGINE/'server-config.json';UNIT='llm-deepseek41-api.service'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('profile',choices=['baseline','gpu-swap16','gpu-heavy3','cache120','prefill128','prefill64']);a=p.parse_args()
    if os.geteuid()!=0:raise SystemExit('Root is required for the authorized service switch')
    with (R/'.benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=CONFIG.read_bytes();target=(R/'profiles'/(a.profile+'.json')).read_bytes()
        allowed={digest(p.read_bytes()) for p in (R/'profiles').glob('*.json')}
        if digest(current) not in allowed:raise RuntimeError('Configuration changed outside this study')
        config=json.loads(target);key=Path(config['api_key_file']).read_text().strip()
        for library in config.get('study_libraries',[]):
            if digest(Path(library['file']).read_bytes())!=library['sha256']:
                raise RuntimeError('Candidate library hash mismatch: '+library['file'])
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url=f"http://{config['host']}:{config['port']}"
        headers={'Authorization':'Bearer '+key}
        def get(endpoint):
            with opener.open(urllib.request.Request(url+endpoint,headers=headers),timeout=5) as res:return json.load(res)
        try:
            slots=get('/slots')
            if any(x.get('is_processing') for x in slots):raise RuntimeError('API request active; retry after it finishes')
        except (urllib.error.URLError,TimeoutError):
            # A missing listener is acceptable only if the service is actually inactive.
            if subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode==0:raise
        stamp=time.strftime('%Y%m%d-%H%M%S');folder=R/'switches'/f'{stamp}-{a.profile}';folder.mkdir(parents=True)
        status={'profile':a.profile,'phase':'switching','previous_config_sha256':digest(current),
                'target_config_sha256':digest(target),'start_unix':time.time(),'error':None,
                'server_log_start_bytes':(ENGINE/'server.stderr.log').stat().st_size}
        def save():
            temp=folder/'STATUS.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(folder/'STATUS.json')
        save()
        spec=importlib.util.spec_from_file_location('validated_switch',LAB/'iq-cache-next-20261103/switch_profile.py')
        switch=importlib.util.module_from_spec(spec);spec.loader.exec_module(switch)
        try:
            switch.apply(current,target)
            status['phase']='loading';save();deadline=time.monotonic()+900
            while True:
                if CONFIG.read_bytes()!=target:raise RuntimeError('Configuration changed while loading')
                if (ENGINE/'STOP_REQUESTED').exists():raise RuntimeError('API protection stop; inspect existing guard before any restart')
                active=subprocess.run(['systemctl','is-active','--quiet',UNIT]).returncode==0
                if not active:raise RuntimeError('API service became inactive during loading')
                try:ready=get('/health').get('status')=='ok'
                except (urllib.error.URLError,TimeoutError):ready=False
                status['last_check_unix']=time.time();status['ready']=ready
                if ready:
                    status['phase']='ready';status['elapsed_s']=time.time()-status['start_unix'];save()
                    print(json.dumps({'saved':str(folder),'profile':a.profile,'ready':True,'elapsed_s':status['elapsed_s']}),flush=True);return
                if time.monotonic()>deadline:raise TimeoutError('Readiness wait expired; live service is left intact for inspection')
                save();time.sleep(10)
        except BaseException as e:
            status['phase']='error';status['error']=repr(e);save();raise
if __name__=='__main__':main()
