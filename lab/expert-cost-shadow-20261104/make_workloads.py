from pathlib import Path
import hashlib,json,random,struct
R=Path(__file__).resolve().parent

def routes(E,T,A,family,layout,seed):
    rng=random.Random(seed)
    active=list(range(A)) if layout=='contiguous' else [i*(E-1)//(A-1) for i in range(A)]
    result=[]
    for t in range(T):
        if family=='uniform':selected=[active[(t*6+j)%A] for j in range(6)]
        elif family=='random':selected=rng.sample(active,6)
        elif family=='skew':selected=active[:6] if rng.random()<.75 else rng.sample(active,6)
        elif family=='zipf':
            selected=[]
            while len(selected)<6:
                value=rng.choices(active,weights=[1/(i+1)**.8 for i in range(A)])[0]
                if value not in selected:selected.append(value)
        elif family=='rotating':
            shift=(t//13)*6
            selected=[active[(shift+j)%A] for j in range(6)] if t%3 else rng.sample(active,6)
        else:raise ValueError(family)
        assert len(set(selected))==6
        result+=selected
    return result

def main():
    # Calibration and final-test workloads are never used for coefficient fitting.
    train=[(24,64,24,'uniform','contiguous'),(48,64,24,'random','spread'),(96,64,48,'uniform','spread'),
      (24,128,12,'skew','spread'),(48,128,48,'random','contiguous'),(96,128,96,'uniform','contiguous'),
      (24,256,24,'uniform','contiguous'),(48,256,24,'skew','spread'),(96,256,96,'random','contiguous'),
      (24,512,12,'random','spread'),(48,512,48,'uniform','contiguous'),(96,512,48,'skew','spread'),
      (24,1024,24,'skew','contiguous'),(48,1024,48,'random','contiguous'),(96,1024,96,'uniform','contiguous'),
      (96,256,12,'uniform','spread'),(96,512,24,'random','spread'),(48,64,6,'uniform','spread'),
      (48,256,48,'skew','contiguous'),(96,1024,96,'skew','spread'),(96,128,12,'random','spread'),
      (24,64,12,'random','contiguous'),(48,512,12,'uniform','spread'),(96,64,96,'random','contiguous'),
      # Explicit CPU tile and GPU BLAS dispatch boundaries; registered before any heldout timings.
      (96,1008,96,'uniform','contiguous'),(48,504,48,'uniform','contiguous'),(24,252,24,'uniform','contiguous'),
      (96,112,96,'uniform','contiguous'),(96,144,96,'uniform','contiguous'),(96,240,96,'uniform','contiguous'),
      (96,272,96,'uniform','contiguous'),(96,80,96,'uniform','contiguous'),(48,520,48,'uniform','contiguous')]
    cal=[(48,192,24,'rotating','spread'),(96,384,48,'rotating','spread'),(24,96,12,'rotating','contiguous'),
         (96,768,96,'rotating','contiguous'),(48,384,48,'rotating','contiguous'),(96,192,12,'rotating','spread')]
    test=[(48,96,24,'zipf','spread'),(96,192,48,'zipf','spread'),(24,384,12,'zipf','contiguous'),
          (96,768,96,'zipf','contiguous'),(48,768,48,'zipf','contiguous'),(96,384,24,'zipf','spread'),
          (24,192,24,'zipf','contiguous'),(96,96,96,'zipf','spread')]
    folder=R/'workloads';folder.mkdir(exist_ok=True);records=[]
    for split,cases in [('train',train),('calibration',cal),('test',test)]:
        for i,(E,T,A,family,layout) in enumerate(cases):
            seed=41000+{'train':0,'calibration':1000,'test':2000}[split]+i
            data=routes(E,T,A,family,layout,seed);raw=struct.pack('<'+'i'*len(data),*data)
            name=f'{split}-{i:02d}';p=folder/(name+'.bin')
            if p.exists():assert p.read_bytes()==raw
            else:p.write_bytes(raw)
            counts=[0]*E
            for e in data:counts[e]+=1
            records.append(dict(id=name,split=split,experts=E,tokens=T,intended_active=A,family=family,layout=layout,seed=seed,
                path=str(p),sha256=hashlib.sha256(raw).hexdigest(),histogram=counts))
    p=R/'WORKLOADS.json';text=json.dumps(records,indent=2)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    plan={'schema':1,'workloads_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'cases':[c['name'] for c in json.loads((R.parent/'expert-validation-20261101/cases.json').read_text())],
          'workloads':[{k:w[k] for k in ['id','split','sha256']} for w in records],
          'support_protocol':'measured E/T/active/maxn; per-target feature ranges; nearest normalized distance <=1.5*training leave-one-out max',
          'target_protocol':'fit train only; error envelopes calibration only; final Zipf family after code/model freeze'}
    q=R/'STUDY.json';pt=json.dumps(plan,indent=2)+'\n'
    if q.exists():assert q.read_text()==pt
    else:q.write_text(pt)
    print('Prepared',len(records),'disjoint workloads')
if __name__=='__main__':main()
