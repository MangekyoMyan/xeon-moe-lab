from pathlib import Path
import json,random,struct,hashlib
R=Path(__file__).resolve().parent;out=[];(R/'workloads').mkdir(exist_ok=True)
for name,T,family,seed in [('smoke',32,'small',51001),('balanced128',128,'balanced',51002),('balanced512',512,'balanced',51003),('zipf128',128,'zipf',51004),('zipf512',512,'zipf',51005),('hot512',512,'hot',51006),('balanced1024',1024,'balanced',51007)]:
    rng=random.Random(seed);ids=[]
    for t in range(T):
        if family=='small':row=rng.sample(range(12),6)
        elif family=='balanced':row=[(t*6+j)%384 for j in range(6)]
        elif family=='hot':row=list(range(6)) if t%4 else rng.sample(range(384),6)
        else:
            row=[]
            while len(row)<6:
                e=rng.choices(range(384),weights=[(i+1)**-1.1 for i in range(384)])[0]
                if e not in row:row.append(e)
        assert len(set(row))==6;ids.extend(row)
    raw=struct.pack('<'+'i'*len(ids),*ids);path=R/'workloads'/(name+'.bin')
    if path.exists():assert path.read_bytes()==raw
    else:path.write_bytes(raw)
    h=[ids.count(e) for e in range(384)]
    out.append({'name':name,'tokens':T,'family':family,'seed':seed,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'histogram':h})
(R/'WORKLOADS.json').write_text(json.dumps(out,indent=2)+'\n')
print('Prepared',len(out),'synthetic routes; no claim of actual production routing')
