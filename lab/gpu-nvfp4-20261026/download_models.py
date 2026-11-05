#!/usr/bin/env python3
import datetime,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
models=['michaelw9999/Qwen3.6-27B-NVFP4-MTP-GGUF','FreedomAISVR/GLM-4.7-Flash-NVFP4-GGUF']
state={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':[]}
def save():
 temp=ROOT/'download-state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ROOT/'download-state.json')
for repo in models:
 meta=json.loads((ROOT/'model-metadata'/(repo.replace('/','--')+'.json')).read_text())
 files=[x for x in meta['siblings'] if x['rfilename'].endswith('.gguf')]
 assert len(files)==1
 entry=files[0];size=entry['size'];expected=entry['lfs']['sha256'];name=entry['rfilename'];assert '/' not in name
 folder=LAB/'models'/repo.replace('/','--');folder.mkdir(exist_ok=True)
 target=folder/name;part=folder/(name+'.part')
 item={'repo':repo,'revision':meta['sha'],'file':name,'bytes':size,'sha256':expected,'path':str(target),'status':'downloading'};state['models'].append(item);save()
 if not target.exists():
  url='https://huggingface.co/'+repo+'/resolve/'+meta['sha']+'/'+name+'?download=true'
  cmd=['curl','--fail','--location','--retry','12','--retry-delay','5','--retry-all-errors','--connect-timeout','30','--speed-time','120','--speed-limit','1024','--limit-rate','30M','--continue-at','-','--output',str(part),url]
  print('Downloading',repo,name,size,flush=True)
  result=subprocess.run(cmd)
  if result.returncode!=0:item['status']='download_failed';item['returncode']=result.returncode;save();raise SystemExit(result.returncode)
  assert part.stat().st_size==size
  check=part
 else:check=target
 item['status']='checking_sha256';save();h=hashlib.sha256()
 with check.open('rb') as f:
  while chunk:=f.read(8<<20):h.update(chunk)
 assert h.hexdigest()==expected,(repo,h.hexdigest(),expected)
 if check==part:part.replace(target)
 item['status']='verified';item['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 (folder/'download-verified.json').write_text(json.dumps(item,indent=2)+'\n');print('Verified',repo,flush=True)
state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
