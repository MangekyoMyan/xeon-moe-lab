#!/usr/bin/env python3
import urllib.request, time, hashlib, json, os
from pathlib import Path
root=Path(__file__).resolve().parent
out=root.parent/'results'/root.name
url='https://download.nvidia.com/XFree86/Linux-x86_64/580.196.07/NVIDIA-Linux-x86_64-580.196.07.run'
target=root/url.rsplit('/',1)[-1]
tmp=target.with_suffix('.run.partial')
h=hashlib.sha256();size=0;last=time.monotonic()
with urllib.request.urlopen(url,timeout=60) as r,tmp.open('wb') as f:
 expected=int(r.headers.get('Content-Length',0))
 while b:=r.read(1<<20):
  f.write(b);h.update(b);size+=len(b)
  if time.monotonic()-last>20:
   print('Downloaded',round(size/1e6),'MB',flush=True);last=time.monotonic()
 f.flush();os.fsync(f.fileno())
assert expected and size==expected
os.replace(tmp,target)
record={'url':url,'bytes':size,'sha256_local':h.hexdigest(),'hash_is_local_fingerprint_not_published_runfile_digest':True}
(out/'driver-runfile-download.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
