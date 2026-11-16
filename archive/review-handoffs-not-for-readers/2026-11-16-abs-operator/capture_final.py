"""Read-only, whitelisted runtime receipt. Never serialize API credentials."""
from pathlib import Path
import hashlib,json,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;LAB=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    config=LAB/'deepseek41-bringup-20261101/server-config.json'
    h=json.loads((config.parent/'health.json').read_text());pid=h.get('pid')
    assert h.get('status')=='running' and pid
    assert (Path('/proc')/str(pid)/'comm').read_text().strip()=='llama-server'
    maps=(Path('/proc')/str(pid)/'maps').read_text().splitlines()
    paths=sorted({row.split()[-1] for row in maps if '/libggml-cpu.so.' in row})
    assert len(paths)==1 and 'release-candidate-a' in paths[0],paths
    libraries={p:sha(Path(p)) for p in paths}
    assert next(iter(libraries.values()))=='33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3'
    assert sha(config)=='963b652592504501b6ec02880ab7e38303b7da782f7aca8e49ff0a9568d70d43'
    key=(LAB/'api-server-20261030/api-key.txt').read_text().strip()
    def request(path):
        req=urllib.request.Request('http://10.10.10.2:8000/'+path,headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
    health=request('health');slots=request('slots')
    services={s:subprocess.run(['systemctl','is-active',s],capture_output=True,text=True,timeout=5).stdout.strip() for s in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']}
    monitor=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    result={'unix':time.time(),'api_pid':pid,'config_sha256':sha(config),'runtime_cpu_library':libraries,'health':{'status':health.get('status')},'slots':[{'id':s.get('id'),'is_processing':s.get('is_processing')} for s in slots],'services':services,'monitor':{k:monitor.get(k) for k in ['unix','cpu','bmc','gpu','host_counters','errors','active_alerts']},'api_restart_by_this_study':False,'production_files_modified':False}
    (R/'FINAL_RUNTIME.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'api_pid':pid,'health':result['health'],'slots':result['slots'],'services':services,'config_and_cpu_library_match_frozen':True,'monitor_errors':monitor.get('errors'),'active_alerts':monitor.get('active_alerts')}))
if __name__=='__main__':main()
