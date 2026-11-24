from pathlib import Path
from contextlib import contextmanager
import hashlib,json,subprocess,time,urllib.request
LAB=Path(__file__).resolve().parent.parent;E=LAB/'deepseek41-bringup-20261101'
def read(p):return json.loads(p.read_text())
def write(p,j):
    temporary=p.with_suffix(".tmp")
    temporary.write_text(json.dumps(j,indent=2)+'\n')
    temporary.replace(p)
def active(u):return subprocess.run(['systemctl','is-active','--quiet',u]).returncode==0
def clear_monitor(root):
    for u in ['xeon-watch.service','xeon-watch-notify.service']:assert active(u),'monitor inactive'
    s=read(Path('/var/lib/xeon-watch/latest.json'));assert 0<=time.time()-s['unix']<=45 and not s['errors'] and not s['active_alerts'],'monitor freshness/alarm'
    for p in [E/'STOP_REQUESTED',LAB/'moe-engine-20261029/STOP_REQUESTED',*root.glob('**/STOP_REQUESTED')]:assert not p.exists(),'hardware stop requires investigation'
@contextmanager
def window(root,label):
    clear_monitor(root);out=root/('API_WINDOW_'+label+'.json');previous=read(out) if out.exists() else {}
    was=active('llm-deepseek41-api.service');inherited=bool(previous.get('restore_required'));cfg=read(E/'server-config.json');digest=hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest();assert digest==read(root/'production-before-live.json')['config_sha256']
    rec=dict(initially_active=was or inherited,restore_required=inherited,config_sha256=digest,phase='preparing',started_unix=time.time());write(out,rec)
    if was:
        req=urllib.request.Request(f"http://{cfg['host']}:{cfg['port']}/slots",headers={'Authorization':'Bearer '+Path(cfg['api_key_file']).read_text().strip()})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=10) as f:slots=json.load(f)
        assert isinstance(slots,list) and slots and all(s.get('is_processing') is False for s in slots),'API request still running'
    attempted=inherited
    try:
        if was:
            attempted=True;rec.update(restore_required=True,phase='stopping API');write(out,rec)
            subprocess.run(['systemctl','stop','llm-deepseek41-api.service'],check=True,timeout=90)
        rec['phase']='experiment';write(out,rec);yield
    finally:
        if attempted:
            try:
                clear_monitor(root);assert hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest()==digest
                if not active('llm-deepseek41-api.service'):
                    deadline=time.monotonic()+60
                    while True:
                        clear_monitor(root);s=read(Path('/var/lib/xeon-watch/latest.json'))
                        if s['gpu']['used_MiB']<=128:break
                        if time.monotonic()>deadline:raise TimeoutError('GPU release not yet reflected in monitor')
                        time.sleep(2)
                    subprocess.run(['python3',str(E/'start.py')],check=True,timeout=60)
                receipt=root/('RESTORED_'+label+'.json')
                subprocess.run(['python3',str(root/'verify_restore.py'),str(receipt)],check=True,timeout=7320)
                rec.update(phase='original API restored',restore_required=False,receipt=str(receipt),finished_unix=time.time());write(out,rec)
            except Exception as e:
                rec.update(phase='restoration requires attention',restore_required=True,error=str(e));write(out,rec);raise
        else:
            rec.update(phase='complete; originally inactive API kept inactive',restore_required=False,finished_unix=time.time());write(out,rec)
