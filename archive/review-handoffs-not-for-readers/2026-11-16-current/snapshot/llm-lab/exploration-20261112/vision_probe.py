"""Bounded local-image API smoke test; daemon supplies hardware samples."""
from pathlib import Path
import argparse,base64,fcntl,hashlib,importlib.util,json,os,threading,time,urllib.request
R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
spec=importlib.util.spec_from_file_location('helpers',R.parent/'postcompact-20261103/live_probe.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);a=parser.parse_args();assert a.label.replace('-','').isalnum()
 lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=(ENGINE/'server-config.json').read_bytes();cfg=json.loads(raw);key=Path(cfg['api_key_file']).read_text().strip();url=f"http://{cfg['host']}:{cfg['port']}";opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
 def get(route):
  with opener.open(urllib.request.Request(url+route,headers=headers),timeout=5) as response:return json.load(response)
 assert get('/health')['status']=='ok' and not any(x.get('is_processing') for x in get('/slots'))
 fixture=ENGINE/'source/tools/mtmd/test-1.jpeg';data=fixture.read_bytes();sha=hashlib.sha256(data).hexdigest();assert sha=='2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3'
 prompt='Read the largest English headline at the top of this newspaper. Reply only with the headline.'
 body={'model':cfg['model_alias'],'messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}}]}],'max_tokens':32,'temperature':0,'seed':42,'stream':True,'stream_options':{'include_usage':True},'cache_prompt':False}
 folder=R/'vision'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True,mode=0o700)
 initial=helpers.watch();pid=helpers.api_pid();record={'label':a.label,'pid':pid,'config_sha256':hashlib.sha256(raw).hexdigest(),'fixture':str(fixture),'fixture_sha256':sha,'question':prompt,'initial_watch':initial,'samples':[],'error':None};stop=threading.Event();failures=[]
 def monitor():
  while not stop.wait(.5):
   try:
    sample=helpers.watch();record['samples'].append(sample)
    if sample['host_counters']!=initial['host_counters']:raise RuntimeError('Hardware counter change')
   except Exception as ex:failures.append(repr(ex));return
 worker=threading.Thread(target=monitor,daemon=True);response=None;pieces=[];first=None;usage=None;finish=None;start=time.monotonic()
 try:
  worker.start();response=opener.open(urllib.request.Request(url+'/v1/chat/completions',headers=headers,data=json.dumps(body).encode()),timeout=180)
  for line in response:
   if failures:raise RuntimeError(failures[0])
   if time.monotonic()-start>180:raise TimeoutError('Vision smoke test time bound')
   line=line.decode().strip()
   if not line.startswith('data:'):continue
   payload=line[5:].strip()
   if payload=='[DONE]':break
   event=json.loads(payload)
   if event.get('error'):raise RuntimeError(str(event['error']))
   if event.get('usage'):usage=event['usage']
   for choice in event.get('choices',[]):
    text=choice.get('delta',{}).get('content') or ''
    if text and first is None:first=time.monotonic()
    pieces.append(text)
    if choice.get('finish_reason'):finish=choice['finish_reason']
  assert finish is not None and (ENGINE/'server-config.json').read_bytes()==raw and helpers.api_pid()==pid
  after=helpers.watch();assert after['host_counters']==initial['host_counters']
  answer=''.join(pieces);record.update(answer=answer,headline_terms_present=all(x in answer.upper() for x in ['MEN','WALK','MOON']),usage=usage,finish_reason=finish,elapsed_s=time.monotonic()-start,ttft_s=None if first is None else first-start,after_watch=after)
 except BaseException as error:record['error']=repr(error);raise
 finally:
  if response:response.close()
  stop.set()
  if worker.is_alive():worker.join(timeout=2)
  samples=[initial,*record['samples']]+([record['after_watch']] if 'after_watch'in record else [])
  record['observed_gpu_peak_MiB']=max(x['gpu']['used_MiB'] for x in samples)
  fd=os.open(folder/'RESULT.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  if os.geteuid()==0:os.fchown(fd,R.stat().st_uid,R.stat().st_gid);os.chown(folder,R.stat().st_uid,R.stat().st_gid)
  with os.fdopen(fd,'w') as out:json.dump(record,out,ensure_ascii=False,indent=2);out.write('\n')
 print(json.dumps({k:record.get(k) for k in ['label','answer','headline_terms_present','usage','ttft_s','observed_gpu_peak_MiB','error']},ensure_ascii=False),flush=True);print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
