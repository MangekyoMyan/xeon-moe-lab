"""Controlled per-process perf stat for decode-only coordination comparisons."""
from pathlib import Path
import json,os,select,signal,subprocess,time
class TaskStat:
 def __init__(self,pid,folder,extra_events=None):
  self.folder=folder;self.controls=[];read,write=os.pipe();ack_read,ack_write=os.pipe();self.write=write;self.ack_read=ack_read
  self.extra_events=list(extra_events or [])
  assert all(x in ['r8408:u','r8208:u'] for x in self.extra_events)
  self.log=(folder/'cpu-stat.stderr').open('w')
  event_string='{'+','.join(['cycles:u','instructions:u','ref-cycles:u',*self.extra_events])+'},task-clock,context-switches,cpu-migrations'
  self.command=['perf','stat','--no-big-num','-x',';','-p',str(pid),'-D','-1','--control',f'fd:{read},{ack_write}','-e',event_string,'-o',str(folder/'cpu-stat.csv')]
  env=os.environ.copy();env['LC_ALL']='C'
  self.process=subprocess.Popen(self.command,pass_fds=(read,ack_write),stdout=self.log,stderr=self.log,env=env);os.close(read);os.close(ack_write)
  try:self.control('disable')
  except BaseException:self.close();raise
 def control(self,command):
  before=time.monotonic();os.write(self.write,(command+'\n').encode())
  if not select.select([self.ack_read],[],[],10)[0]:raise RuntimeError('CPU perf stat control timeout')
  ack=os.read(self.ack_read,256)
  if b'ack' not in ack:raise RuntimeError('CPU perf stat ACK absent')
  self.controls.append({'command':command,'before':before,'after':time.monotonic()})
 def close(self):
  requested_sigint=False;forced_kill=False
  if self.process.poll() is None:
   requested_sigint=True
   self.process.send_signal(signal.SIGINT)
   try:self.process.wait(timeout=10)
   except subprocess.TimeoutExpired:forced_kill=True;self.process.kill();self.process.wait()
  os.close(self.write);os.close(self.ack_read);self.log.close()
  result={'command':self.command,'controls':self.controls,'exit_code':self.process.returncode,'events':[],
          'requested_sigint':requested_sigint,'forced_kill':forced_kill,
          'scope':'Observed client decode window, from perf enable to disable; not the exact server timing interval'}
  path=self.folder/'cpu-stat.csv'
  if path.exists():
   for line in path.read_text().splitlines():
    if not line or line.startswith('#'):continue
    columns=line.split(';')
    if len(columns)<3:continue
    row={'raw':line,'event':columns[2].strip(),'unit':columns[1].strip()}
    try:row['value']=float(columns[0].strip())
    except ValueError:row['value']=None
    if len(columns)>3:
     try:row['event_runtime_ns']=float(columns[3].strip())
     except ValueError:pass
    if len(columns)>4:
     try:row['running_percent']=float(columns[4].strip())
     except ValueError:pass
    result['events'].append(row)
  errors=[]
  intentional_signal_exit=requested_sigint and not forced_kill and result['exit_code']==-signal.SIGINT
  if result['exit_code']!=0 and not intentional_signal_exit:errors.append('perf exited unexpectedly')
  required={'cycles:u','instructions:u','ref-cycles:u','task-clock','context-switches','cpu-migrations',*self.extra_events}
  for event in required:
   rows=[x for x in result['events'] if x['event']==event]
   if len(rows)!=1 or rows[0]['value'] is None:
    errors.append('missing, repeated, or uncounted event: '+event);continue
   row=rows[0]
   if row.get('running_percent',0)<99:errors.append('event running percentage below99: '+event)
   if event in ['cycles:u','instructions:u','ref-cycles:u','task-clock'] and row['value']<=0:
    errors.append('nonpositive work counter: '+event)
  starts=[x for x in self.controls if x['command']=='enable']
  ends=[x for x in self.controls if x['command']=='disable']
  if len(starts)==1 and len(ends)==2:
   result['observed_window_seconds_bounds']=[ends[-1]['before']-starts[0]['after'],ends[-1]['after']-starts[0]['before']]
  else:errors.append('unexpected enable/disable sequence')
  result.update(valid=not errors,validation_errors=errors)
  (self.folder/'CPU_STAT.json').write_text(json.dumps(result,indent=2)+'\n')

def process_cpu(pid):
 # /proc/TGID/stat aggregates CPU time across the thread group, including exited workers.
 values=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();hz=os.sysconf('SC_CLK_TCK')
 return {'user_seconds':int(values[11])/hz,'system_seconds':int(values[12])/hz,'clock_ticks':hz,'monotonic':time.monotonic()}

def thread_cpu(pid):
 rows=[];hz=os.sysconf('SC_CLK_TCK');started=time.monotonic()
 for task in sorted(Path(f'/proc/{pid}/task').iterdir(),key=lambda x:int(x.name)):
  try:
   values=(task/'stat').read_text().rsplit(')',1)[1].split()
   rows.append({'tid':int(task.name),'cpu_seconds':(int(values[11])+int(values[12]))/hz,'last_cpu':int(values[36]),'affinity':sorted(os.sched_getaffinity(int(task.name)))})
  except (FileNotFoundError,ProcessLookupError):continue
 return {'start_monotonic':started,'end_monotonic':time.monotonic(),'threads':rows}

def omp_environment(pid):
 env={}
 for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
  key,separator,value=item.partition(b'=')
  if separator and key.startswith((b'OMP_',b'GOMP_')):env[key.decode()]=value.decode()
 return env
