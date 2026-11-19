"""Bounded readiness wait and backing-file check of the restored ordinary API."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import sys
import urllib.request

R=Path(__file__).resolve().parent
E=R.parent/'deepseek41-bringup-20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=Path(sys.argv[1]);assert out.parent==R and not out.exists()
    old=json.loads((R.parent/'route37-integration-20261117/RESTORED_RUNTIME.json').read_text())
    cfg=json.loads((E/'server-config.json').read_text())
    assert sha(E/'server-config.json')==old['config_sha256']
    req=urllib.request.Request('http://'+cfg['host']+':'+str(cfg['port'])+'/health',headers={'Authorization':'Bearer '+Path(cfg['api_key_file']).read_text().strip()})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+600
    while True:
        try:
            with opener.open(req,timeout=5) as f:health=json.load(f)
            if health.get('status')=='ok':break
        except (OSError,ValueError):pass
        if (E/'STOP_REQUESTED').exists() or subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode:raise RuntimeError('Restore stopped; inspect existing guard')
        if time.monotonic()>deadline:raise TimeoutError('Readiness wait exceeded; supervisor remains in control')
        time.sleep(10)
    guard=json.loads((E/'health.json').read_text());pid=guard['pid']
    maps=Path('/proc/'+str(pid)+'/maps').read_text().splitlines()
    all_paths={line.split()[-1] for line in maps if len(line.split())>=6 and line.split()[-1].startswith('/')}
    paths={path for path in all_paths if '/libggml' in path or '/libllama' in path}
    expected=old['mapped_runtime_hashes']
    target_names={Path(path).name for path in expected}
    actual_targets={path for path in paths if Path(path).name in target_names}
    assert actual_targets==set(expected),'normal runtime has extra or missing target libraries'
    assert not any('route37' in path or '/prefix-engine/' in path or 'libtrace169' in path or '/169a-exploration-20261119/' in path for path in all_paths),'experimental runtime remains mapped'
    for p,digest in expected.items():assert sha(Path(p))==digest
    monitor=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    assert 0<=time.time()-monitor['unix']<=45 and not monitor['errors'] and not monitor['active_alerts']
    services={name:subprocess.check_output(['systemctl','is-active',name],text=True).strip() for name in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']}
    result=dict(unix=time.time(),pid=pid,API_health='ok',config_sha256=old['config_sha256'],mapped_runtime_hashes=expected,
                normal_runtime_backing_files_verified=True,diagnostic_libraries_loaded=False,services=services,
                monitor_errors=monitor['errors'],active_alerts=monitor['active_alerts'],monitor_timestamp=monitor['unix'],normal_API_restored=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
