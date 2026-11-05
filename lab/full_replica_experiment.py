#!/usr/bin/env python3
"""Isolated single-inference NUMA replication measurements for the local 26B model."""
import argparse,collections,ctypes,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path
from bench_deepseek_light import snapshot
LAB=Path(__file__).resolve().parent
ROOT=LAB/'review-full-replica-20261020'; OUT=LAB/'results/full-replica-20261020'
MODEL=next((LAB/'models').rglob('Melody*.gguf'))
BUILD=ROOT/'build'; ACTIVE=LAB/'build-avx-prefetch'
BASE_ENV=dict(os.environ,OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',GGML_CPU_NUMA_REPLICA='0',GGML_CPU_NUMA_REPLICA_TRACE='0')
for k in ['LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','GOMP_SPINCOUNT','OMP_WAIT_POLICY']:
 if os.environ.get(k): raise RuntimeError('Unset '+k)

def run(label,mode,threads=24,prompt=0,tokens=128,repetitions=3,depth=0,trace=False,build=BUILD,extra=None,bench_extra=None):
 d=OUT/label;d.mkdir()
 env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode),GGML_CPU_NUMA_REPLICA_TRACE=str(int(trace)))
 if extra: env.update(extra)
 cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(build/'bin/llama-bench'),'-m',str(MODEL),'-p',str(prompt),'-n',str(tokens),'-d',str(depth),'-t',str(threads),'-ngl','0','-r',str(repetitions),'-o','json','--offline','--progress','-b','512','-ub','512','-lm','mmap','--cpu-mask','0xffffff','--cpu-strict','1']
 if bench_extra: cmd.extend(bench_extra)
 rec={'command':cmd,'mode':mode,'parent_thp_enabled':next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('THP_enabled:')),'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','OMP_','OPENBLAS_'))},'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'samples':[],'phases':[]}
 print(json.dumps({'start':label,'mode':mode,'threads':threads}),flush=True)
 start=time.monotonic(); captured=False; rollup=False;seen=set()
 with (d/'bench.json').open('w') as out,(d/'bench.log').open('w') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,stdin=subprocess.DEVNULL,env=env)
  try:
   while child.poll() is None:
    s=snapshot(child.pid)
    if s:
     rec['samples'].append(s)
     if len(rec['samples']) % 30 == 1:
      masks=collections.Counter()
      for f in Path(f'/proc/{child.pid}/task').glob('*/status'):
       try:
        masks[next(x.split(':',1)[1].strip() for x in f.read_text().splitlines() if x.startswith('Cpus_allowed_list:'))]+=1
       except (OSError,StopIteration):pass
      s['thread_affinity_masks']=dict(masks)
    log=(d/'bench.log').read_text()
    for line in log.splitlines():
     if 'llama-bench: benchmark' in line and line not in seen:
      seen.add(line);rec['phases'].append({'time':time.time(),'message':line}); print(json.dumps({'label':label,'phase':line}),flush=True)
    if not trace and not rollup and 'generation run 1/' in log:
     (d/'generation-status.txt').write_text(Path(f'/proc/{child.pid}/status').read_text())
     rollup=True
    if trace and not captured and 'generation run 1/1' in log:
     for f in ['maps','numa_maps','status','smaps_rollup']:
      (d/('proc-'+f+'.txt')).write_text(Path(f'/proc/{child.pid}/{f}').read_text())
     query_pages(child.pid,log,d)
     captured=True
    if time.monotonic()-start>1800: raise TimeoutError(label)
    time.sleep(1)
  finally:
   if child.poll() is None: child.terminate()
   try:rec['exit_code']=child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();rec['exit_code']=child.wait()
 rec['wall_seconds']=time.monotonic()-start
 if rec['exit_code']==0:rec['benchmarks']=json.loads((d/'bench.json').read_text())
 rec['max_rss_gib']=max((s['hwm_kib'] for s in rec['samples']),default=0)/2**20
 rec['max_disk_read_bytes']=max((s['disk_read_bytes'] for s in rec['samples']),default=0)
 rec['cpu_library_sha256']=hashlib.sha256((build/'bin/libggml-cpu.so').read_bytes()).hexdigest()
 (d/'summary.json').write_text(json.dumps(rec,indent=2))
 if rec['exit_code']:raise RuntimeError((d/'bench.log').read_text()[-4000:])
 print(json.dumps({'done':label,'rates':[{'threads':b['n_threads'],'prompt':b['n_prompt'],'tokens':b['n_gen'],'ts':b['avg_ts']} for b in rec['benchmarks']],'max_rss_gib':rec['max_rss_gib'],'disk':rec['max_disk_read_bytes']}),flush=True)
 if trace: placement(d,mode)
 return rec

def query_pages(pid,log,d):
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 maps=re.findall(r'NUMA_REPLICA_MAP node=(\d+) address=(0x[0-9a-f]+) bytes=(\d+)',log)
 if not maps:return
 base=int(maps[0][1],16);records=[]
 for name,size,offset in re.findall(r'NUMA_REPLICA_TENSOR (\S+) bytes=(\d+) offset=(\d+)',log):
  if name not in ['blk.0.ffn_gate_up_exps.weight','blk.0.ffn_down_exps.weight','token_embd.weight']:continue
  start=(base+int(offset)+4095)//4096*4096
  count=min(1024,(base+int(offset)+int(size)-start)//4096)
  addresses=(ctypes.c_void_p*count)(*[start+i*4096 for i in range(count)])
  status=(ctypes.c_int*count)()
  rc=libc.syscall(ctypes.c_long(279),ctypes.c_int(pid),ctypes.c_ulong(count),ctypes.cast(addresses,ctypes.c_void_p),ctypes.c_void_p(),ctypes.cast(status,ctypes.c_void_p),ctypes.c_int(0))
  if rc<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
  nodes=list(status);assert all(n in (0,1) for n in nodes)
  runs=[]
  for n in nodes:
   if runs and runs[-1]['node']==n:runs[-1]['pages']+=1
   else:runs.append({'node':n,'pages':1})
  records.append({'tensor':name,'address':hex(start),'page_bytes':4096,'nodes':nodes,'node_counts':dict(collections.Counter(nodes)),'runs':runs})
 (d/'page-node-query.json').write_text(json.dumps(records,indent=2))

def placement(d,mode):
 from deepseek_tensors import tensors
 log=(d/'bench.log').read_text();maps=[]
 for line in (d/'proc-maps.txt').read_text().splitlines():maps.append(tuple(int(v,16) for v in line.split()[0].split('-')))
 numa={int(line.split()[0],16):line for line in (d/'proc-numa_maps.txt').read_text().splitlines()}
 regions=[]
 for n,a,z in re.findall(r'NUMA_REPLICA_MAP node=(\d+) address=(0x[0-9a-f]+) bytes=(\d+)',log):
  node,addr,size=int(n),int(a,16),int(z)
  matches=[(lo,hi) for lo,hi in maps if lo<=addr and hi>=addr+size]
  assert len(matches)==1,(n,a,z,matches)
  line=numa[matches[0][0]];pages={int(n):int(c) for n,c in re.findall(r'\bN(\d+)=(\d+)',line)}
  regions.append({'node':node,'bytes':size,'line':line,'pages':pages,'local':pages.get(node,0)>0 and all(n==node or c==0 for n,c in pages.items()),'fully_resident':sum(pages.values())*4096>=size})
 weights=[{'name':n,'bytes':int(z),'offset':int(o)} for n,z,o in re.findall(r'NUMA_REPLICA_TENSOR (\S+) bytes=(\d+) offset=(\d+)',log)]
 original=tensors(MODEL);names={x['name'] for x in weights};missing=[x for x in original if x['name'] not in names]
 calls=[dict(node0=int(a),node1=int(b),fallback=int(c)) for a,b,c in re.findall(r'NUMA_REPLICA_CALLS node0=(\d+) node1=(\d+) fallback=(\d+)',log)]
 rec={'regions':regions,'weights':weights,'gguf_tensors':len(original),'missing_weights':missing,'calls':calls,'all_local':bool(regions) and all(r['local'] for r in regions),'all_resident':bool(regions) and all(r['fully_resident'] for r in regions),'replicated_bytes':sum(r['bytes'] for r in regions),'weight_bytes':sum(x['bytes'] for x in original)}
 (d/'placement.json').write_text(json.dumps(rec,indent=2))
 print(json.dumps({k:v for k,v in rec.items() if k not in ['regions','weights']}),flush=True)
 assert rec['all_resident'] and not missing
 if mode==1:assert rec['all_local']
 if mode==2:assert all(0.45 < r['pages'].get(0,0)/sum(r['pages'].values()) < 0.55 for r in regions)
 assert sum(x['node0'] for x in calls)>0 and sum(x['node1'] for x in calls)>0 and not sum(x['fallback'] for x in calls)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','sweep','confirm','control','placement-control','no-thp','pages-large','pages-fine','prefill']);a=p.parse_args()
 if a.stage=='smoke':run('placement-verified',1,tokens=32,repetitions=1,trace=True)
 elif a.stage=='placement-control':run('placement-control',2,tokens=32,repetitions=1,trace=True)
 elif a.stage in ['pages-large','pages-fine']:
  run(a.stage,2,tokens=64,repetitions=1,trace=True)
 elif a.stage=='sweep':
  for label,mode in [('A',0),('B',1),('C',2)]:run('sweep-'+label,mode,threads='12,24',tokens=64,repetitions=2)
 elif a.stage=='confirm':
  for i,label in enumerate('ABBA'+'BAAB'):
   run(f'confirm-{i}-{label}',2 if label=='A' else 1,tokens=128,repetitions=3)
 elif a.stage=='control':
  for i,label in enumerate('ABBA'):
   run(f'control-{i}-{label}',0 if label=='A' else 1,tokens=128,repetitions=3)
 elif a.stage=='no-thp':
  assert 'THP_enabled:\t0' in Path('/proc/self/status').read_text()
  for i,label in enumerate('ABBA'):
   run(f'no-thp-{i}-{label}',2 if label=='A' else 1,tokens=128,repetitions=2)
 elif a.stage=='prefill':
  for i,label in enumerate('ABBA'):
   run(f'prefill-{i}-{label}',0 if label=='A' else 1,prompt=256,tokens=0,repetitions=2)
