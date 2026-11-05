"""Read saved failure evidence; no GPU polling, settings, service or model mutations."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,re,subprocess,time
R=Path(__file__).resolve().parent;LAB=R.parent;E=LAB/'deepseek41-bringup-20261101';ZONE=ZoneInfo('Asia/Tokyo')
def iso(t):return datetime.fromtimestamp(t,ZONE).isoformat() if isinstance(t,(int,float)) else t
def save(name,obj):
 p=R/name;p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');p.chmod(0o644)
h=json.loads((E/'health.json').read_text());save('health-evidence.json',h)
start=h.get('started_unix',time.time()-3600);finish=h.get('finished_unix',time.time())
summary={k:v for k,v in h.items() if k not in ['recent_samples','latest_bmc','initial_gpu','initial_fans','limits']}
summary['started_JST']=iso(start);summary['finished_JST']=iso(finish)
summary['recent_samples_count']=len(h.get('recent_samples',[]))
summary['recent_samples']=[{'JST':iso(s.get('unix')),'gpu':s.get('gpu'),'cpu_C':s.get('cpu_C'),'available_kib':s.get('available_kib'),'fans':s.get('fans')} for s in h.get('recent_samples',[])]
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();summary['current_boot_id']=boot
for name in ['server.stdout.log','server.stderr.log']:
 p=E/name
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4*1024*1024));lines=f.read().decode(errors='replace').splitlines()
 # Startup and error lines only; do not retain prompts or HTTP request bodies.
 pattern=r'^(?:load_tensors|llama_model_load|llama_model_loader: - type|llama_context|llama_memory|IQ cache:|expert DMA:|ggml_cuda_host_|ggml_backend_cuda|ggml_cuda_init|srv\s+init|main:|CUDA error|error:|error |terminate called|what\(\)|ggml_abort|GGML_ASSERT|Aborted|signal_handler|\.{5,})'
 selected=[x[:1600] for x in lines if re.search(pattern,x)]
 save(name+'.filtered.json',{'file_mtime_JST':iso(p.stat().st_mtime),'selected_lines':selected[-220:]})
start_range=max(start-60,finish-7200)
windows=['--since','@'+str(int(start_range)),'--until','@'+str(int(finish+60))]
for name,args in [('unit-journal',['-u','llm-deepseek41-api.service']),('kernel-journal',['-k'])]:
 p=subprocess.run(['journalctl','-b','0',*args,*windows,'--no-pager','-o','json'],capture_output=True,text=True,timeout=15)
 rows=[]
 for line in p.stdout.splitlines():
  try:j=json.loads(line)
  except json.JSONDecodeError:continue
  msg=j.get('MESSAGE','')
  if not isinstance(msg,str):continue
  if name=='kernel-journal' and not re.search(r'NVRM|Xid|nvidia|oom|Out of memory|Killed process|page allocation failure|segfault|general protection|AER:|PCIe Bus Error|EDAC|MCE|Machine check|watchdog|hung task|blocked for more',msg,re.I):continue
  rows.append({'JST':iso(int(j['__REALTIME_TIMESTAMP'])/1e6),'message':msg[:2000]})
 save(name+'.json',{'returncode':p.returncode,'stderr':p.stderr,'window_start_JST':iso(start_range),'window_end_JST':iso(finish+60),'entries':rows})
print(json.dumps({'health_status':h.get('status'),'error':h.get('error'),'child_returncode':h.get('child_returncode'),'start':iso(start),'finish':iso(finish),'samples':len(h.get('recent_samples',[])),'boot_matches':h.get('boot_id')==boot},ensure_ascii=False))
units={}
for name in ['llm-deepseek41-api.service','xeon-watch.service','xeon-watch-notify.service']:
 p=subprocess.run(['systemctl','show',name,'--property=ActiveState,SubState,Result,MainPID,ExecMainStatus'],capture_output=True,text=True,timeout=5)
 units[name]=dict(s.split('=',1) for s in p.stdout.splitlines() if '=' in s)
summary['units']=units;summary['stop_marker_exists']=(E/'STOP_REQUESTED').exists()
if summary['stop_marker_exists']:summary['stop_marker']=(E/'STOP_REQUESTED').read_text()[:1000]
summary['configuration_sha256']=hashlib.sha256((E/'server-config.json').read_bytes()).hexdigest()
save('SUMMARY_EVIDENCE.json',summary)
print(json.dumps({'units':units,'stop_marker':summary.get('stop_marker'),'evidence':str(R)},ensure_ascii=False))
