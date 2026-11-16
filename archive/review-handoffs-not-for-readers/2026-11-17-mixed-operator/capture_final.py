"""One read-only final runtime check; no service changes or key serialization."""
from pathlib import Path
import hashlib,json,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;LAB=R.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    initial=json.loads((R/'INITIAL_RUNTIME.json').read_text())
    hashes={name:sha(LAB/name) for name in initial['files']}
    assert hashes==initial['files'],'baseline/library/plan/config changed'
    h=json.loads((LAB/'deepseek41-bringup-20261101/health.json').read_text());pid=h.get('pid')
    assert h.get('status')=='running' and pid==initial['health']['pid']
    proc=Path('/proc')/str(pid);assert (proc/'comm').read_text().strip()=='llama-server'
    cpu=sorted({x.split()[-1] for x in (proc/'maps').read_text().splitlines() if '/libggml-cpu.so.' in x})
    assert cpu==[str(LAB/'exploration-20261112/mx-prepack/release-candidate-a/libggml-cpu.so.0.28.0')]
    key=(LAB/'api-server-20261030/api-key.txt').read_text().strip()
    req=urllib.request.Request('http://10.10.10.2:8000/health',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=5) as response:health=json.load(response)
    snapshot=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    services={name:subprocess.run(['systemctl','is-active',name],capture_output=True,text=True,timeout=5).stdout.strip() for name in initial['services']}
    result={'unix':time.time(),'api_pid':pid,'same_pid_as_initial':True,'API_health':health.get('status'),'unchanged_hashes':hashes,'loaded_cpu_library':cpu,'services':services,'monitor':{k:snapshot.get(k) for k in ['unix','cpu','host_counters','errors','active_alerts']},'API_restarted':False,'production_modified':False}
    (R/'FINAL_RUNTIME.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'API_health':result['API_health'],'same_pid':True,'unchanged_hashes':True,'services':services,'monitor_errors':snapshot.get('errors'),'active_alerts':snapshot.get('active_alerts')}))
if __name__=='__main__':main()
