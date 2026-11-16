"""One read-only identity receipt; do not persist secrets or request text."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,time,urllib.request
R=Path(__file__).resolve().parent;LAB=R.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    cfgpath=LAB/'deepseek41-bringup-20261101/server-config.json';cfg=json.loads(cfgpath.read_text())
    health=json.loads((cfgpath.parent/'health.json').read_text());pid=health.get('pid')
    assert health.get('status')=='running' and pid
    proc=Path('/proc')/str(pid);assert (proc/'comm').read_text().strip()=='llama-server'
    maps=(proc/'maps').read_text().splitlines();libraries={}
    for line in maps:
        fields=line.split();name=fields[-1]
        if name.startswith('/') and (Path(name).name.startswith('libggml') or Path(name).name.startswith('libllama')):
            p=Path(name);st=p.stat();dev=fields[3].split(':')
            assert st.st_ino==int(fields[4]) and (os.major(st.st_dev),os.minor(st.st_dev))==tuple(int(v,16) for v in dev)
            libraries[name]={'sha256':sha(p),'inode':st.st_ino,'bytes':st.st_size}
    raw=(proc/'stat').read_text();statfields=raw[raw.rfind(')')+2:].split()
    env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
    allowed=['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','OMP_DYNAMIC','OMP_THREAD_LIMIT','OMP_MAX_ACTIVE_LEVELS','GGML_OP_OFFLOAD_MIN_BATCH','GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH','GGML_CPU_IQ3_COMPACT_MIB','GGML_CPU_IQ_PLAN']
    key=(LAB/'api-server-20261030/api-key.txt').read_text().strip()
    def request(path):
        req=urllib.request.Request(f'http://{cfg["host"]}:{cfg["port"]}/'+path,headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=5) as resp:return json.load(resp)
    api=request('health');slots=request('slots')
    monitor=json.loads(Path('/var/lib/xeon-watch/latest.json').read_text())
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    processes=[]
    if query.returncode==0:
        for row in csv.reader(query.stdout.splitlines()):
            if row:processes.append({'pid':int(row[0].strip()),'gpu_uuid':row[1].strip(),'used_MiB_raw':row[2].strip()})
    services={x:subprocess.run(['systemctl','is-active',x],capture_output=True,text=True,timeout=5).stdout.strip() for x in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']}
    result={'unix':time.time(),'pid':pid,'start_ticks_since_boot':int(statfields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'API_health':api.get('status'),'slots':[{'id':s.get('id'),'is_processing':s.get('is_processing')} for s in slots],'config_sha256':sha(cfgpath),'plan_sha256':sha(LAB/'exploration-20261112/plans/frequency19200.plan'),'mapped_file_identity':libraries,'mapped_file_hash_scope':'Backing path inode/device checked against maps; not a relocated in-memory ELF digest.','executable_sha256':sha(proc/'exe'),'environment':{k:env[k] for k in allowed if k in env},'configuration':{k:cfg.get(k) for k in ['context','slot_count','kv_type','gpu_resident_expert_layers']},'GPU_compute_process_query':{'returncode':query.returncode,'stderr':query.stderr[:4096],'processes':processes},'services':services,'monitor':{k:monitor.get(k) for k in ['unix','cpu','gpu','bmc','host_counters','errors','active_alerts']},'fan_policy':'Preserve max70%; no fan write. RPM is in monitor; duty not independently queried.','actions':'Read-only; no model load, build, API request for generation, settings change or restart.'}
    with (R/'IDENTITY_SNAPSHOT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'pid':pid,'health':result['API_health'],'slots':result['slots'],'libraries':libraries,'GPU_processes':processes,'services':services,'config_sha256':result['config_sha256']}))
if __name__=='__main__':main()
