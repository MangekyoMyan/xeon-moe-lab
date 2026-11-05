#!/usr/bin/env python3
import collections,json,math,struct,sys
from pathlib import Path
p=Path(sys.argv[1]);f=p.open('rb')
def unpack(fmt):
 data=f.read(struct.calcsize('<'+fmt));assert len(data)==struct.calcsize('<'+fmt);v=struct.unpack('<'+fmt,data);return v[0] if len(v)==1 else v
def string():
 n=unpack('Q');assert n<64<<20;data=f.read(n);assert len(data)==n;return data.decode('utf8',errors='replace')
def value(kind):
 fmt={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
 if kind in fmt:return unpack(fmt[kind])
 if kind==8:return string()
 if kind==9:
  subtype,n=unpack('IQ');assert n<10_000_000
  for _ in range(n):value(subtype)
  return {'array_type':subtype,'count':n}
 raise ValueError(kind)
magic,version,nt,nkv=unpack('IIQQ');assert magic==0x46554747 and version==3 and nt<100000 and nkv<10000
meta={}
for _ in range(nkv):key=string();meta[key]=value(unpack('I'))
tensors=[]
for _ in range(nt):
 name=string();nd=unpack('I');assert 1<=nd<=4;shape=[unpack('Q') for _ in range(nd)];kind,offset=unpack('IQ');tensors.append({'name':name,'shape':shape,'type_id':kind,'offset':offset,'elements':math.prod(shape)})
summary={'path':str(p),'file_bytes':p.stat().st_size,'header_bytes':f.tell(),'metadata':meta,'tensors':tensors,'type_tensor_counts':dict(collections.Counter(t['type_id'] for t in tensors)),'type_element_counts':{str(k):sum(t['elements'] for t in tensors if t['type_id']==k) for k in set(t['type_id'] for t in tensors)}}
out=Path(sys.argv[2]);out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['metadata','tensors']},indent=2));print(json.dumps({k:v for k,v in meta.items() if k.startswith(('general.','qwen35.','glm4moe.','glm4moelite.'))},indent=2))
