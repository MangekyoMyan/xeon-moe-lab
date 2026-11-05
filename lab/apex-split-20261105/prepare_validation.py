from pathlib import Path
import hashlib,json,random,struct
R=Path(__file__).resolve().parent
if (R/'VALIDATION_PLAN.json').exists():raise SystemExit('Validation already frozen')
workloads=json.loads((R/'WORKLOADS.json').read_text());new=[]
for name,T,family,seed in [('vbalanced128',128,'balanced',52101),('vbalanced512',512,'balanced',52102),('vzipf512',512,'zipf',52103)]:
    rng=random.Random(seed);permutation=list(range(384));rng.shuffle(permutation);ids=[]
    for t in range(T):
        if family=='balanced':row=[(t*6+j)%384 for j in range(6)]
        else:
            row=[]
            while len(row)<6:
                e=rng.choices(range(384),weights=[(i+1)**-1.1 for i in range(384)])[0]
                if e not in row:row.append(e)
        ids.extend(permutation[e] for e in row)
    raw=struct.pack('<'+'i'*len(ids),*ids);path=R/'workloads'/(name+'.bin');path.write_bytes(raw)
    w={'name':name,'tokens':T,'family':family,'seed':seed,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'histogram':[ids.count(e) for e in range(384)],'physical_expert_ids':'random permutation; high-load GPU experts fragmented'}
    new.append(w);workloads.append(w)
(R/'WORKLOADS.json').write_text(json.dumps(workloads,indent=2)+'\n')
fractions={24:{'vbalanced128':.25,'vbalanced512':.5,'vzipf512':.25},4:{'vbalanced128':.5,'vbalanced512':.9,'vzipf512':.5}}
jobs=[]
for layer in [24,4]:
    for w in new:
        A=sum(n>0 for n in w['histogram']);fraction=fractions[layer][w['name']]
        jobs.append({'layer':layer,'workload':w['name'],'chunk':16,'threads':24,'repeats':7,'gpu_experts':int(A*fraction),'fraction':fraction,'active':A,'route_sha256':w['sha256']})
for layer in [24,4]:
    base=next(x for x in jobs if x['layer']==layer and x['workload']=='vzipf512');jobs.append(dict(base,chunk=32))
plan={'schema':1,'purpose':'Fixed splits chosen before fresh validation timing; not trained/deployed adaptive policy',
      'basis':'pilot chooses25/50/90percent representative points; IQ balanced512 intentionally tests the weak/near-tie case',
      'build':json.loads((R/'BUILD.json').read_text()),'jobs':jobs}
(R/'VALIDATION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(jobs,indent=2))
