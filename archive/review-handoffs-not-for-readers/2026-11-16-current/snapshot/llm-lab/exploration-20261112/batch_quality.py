"""Inspect the synthetic Japanese request that changed after changing batch size."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,threading,time,urllib.request
R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
spec=importlib.util.spec_from_file_location('helpers',R.parent/'postcompact-20261103/live_probe.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();assert a.label.replace('-','').isalnum()
 lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=(ENGINE/'server-config.json').read_bytes();c=json.loads(raw);source=next((R/'runs').glob('*-ubtest-gpu-sort-candidate-r1-japanese-plain-c4096-n16/RESULT.json'));original=json.loads(source.read_text());body=original['request'].copy();body.update(n_probs=20,post_sampling_probs=False,return_tokens=True,stream=False)
 profile=c['optimization_study']['profile'];expected_files=list((R/'runs').glob('*-ubtest-'+profile+'-r1-japanese-plain-c4096-n16/RESULT.json'));assert len(expected_files)==1;expected=json.loads(expected_files[0].read_text())
 key=Path(c['api_key_file']).read_text().strip();url=f"http://{c['host']}:{c['port']}";headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(urllib.request.Request(url+'/slots',headers=headers),timeout=5) as response:assert not any(s.get('is_processing') for s in json.load(response))
 folder=R/'batch-quality'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True,mode=0o700)
 if os.geteuid()==0:os.chown(folder,R.stat().st_uid,R.stat().st_gid)
 initial=helpers.watch();record={'label':a.label,'profile':profile,'config_sha256':hashlib.sha256(raw).hexdigest(),'prompt_source':str(source),'expected_without_probabilities':expected['output_sha256'],'initial_watch':initial,'samples':[],'error':None};stop=threading.Event();failures=[]
 def monitor():
  while not stop.wait(1):
   try:
    state=helpers.watch();record['samples'].append(state)
    if state['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counter change')
   except Exception as ex:failures.append(repr(ex));return
 thread=threading.Thread(target=monitor,daemon=True);response=None;text=[];ids=[];probabilities=[];final=None;start=time.monotonic()
 try:
  thread.start();response=opener.open(urllib.request.Request(url+'/completion',headers=headers,data=json.dumps(body).encode()),timeout=180)
  final=json.load(response)
  if final.get('error'):raise RuntimeError(str(final['error']))
  if failures:raise RuntimeError(failures[0])
  probabilities=final.get('completion_probabilities') or []
  ids=final.get('tokens') or []
  text=[final.get('content','')]
  record.update(raw_completion=final,probability_count=len(probabilities),returned_token_count=len(ids))
  assert final is not None and len(ids)==16 and 0<len(probabilities)<=16
  cursor=0;positions=[]
  for probability in probabilities:
   while cursor<len(ids) and ids[cursor]!=probability['id']:cursor+=1
   assert cursor<len(ids),'Probability records do not match generated token sequence'
   positions.append(cursor);cursor+=1
  record.update(probability_token_indices=positions,missing_probability_token_indices=[i for i in range(len(ids)) if i not in positions],probability_note='Native API can merge UTF8 display text; missing token logprobs are not invented.')
  assert (ENGINE/'server-config.json').read_bytes()==raw
  output=''.join(text);sha=hashlib.sha256(output.encode()).hexdigest();record.update(output=output,output_sha256=sha,token_ids=ids,probabilities=probabilities,matches_uninstrumented_output=sha==expected['output_sha256'],timings=final['timings'],elapsed_s=time.monotonic()-start)
 except BaseException as ex:record['error']=repr(ex);raise
 finally:
  if response:response.close()
  stop.set()
  if thread.is_alive():thread.join(timeout=2)
  fd=os.open(folder/'RESULT.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  if os.geteuid()==0:os.fchown(fd,R.stat().st_uid,R.stat().st_gid)
  with os.fdopen(fd,'w') as out:json.dump(record,out,ensure_ascii=False,indent=2);out.write('\n')
 print(json.dumps({'label':a.label,'output':record['output'],'matches_uninstrumented_output':record['matches_uninstrumented_output'],'saved':str(folder)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
