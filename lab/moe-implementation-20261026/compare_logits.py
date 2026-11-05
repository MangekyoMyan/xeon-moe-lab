#!/usr/bin/env python3
"""Stream full vocabulary comparisons; diagnostic, never run during timing."""
import argparse,array,hashlib,json,math,struct
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('reference',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
assert not a.output.exists()
def header(f):
    raw=f.read(12);v,n,p=struct.unpack('iii',raw);ids=f.read(4*(p+n-1));return v,n,p,ids
def row(f,v):
    pos,top=struct.unpack('ii',f.read(8));x=array.array('f');x.frombytes(f.read(4*v));assert len(x)==v and all(map(math.isfinite,x));return pos,top,x
result={'reference':str(a.reference),'candidate':str(a.candidate),'scope':'Fixed teacher-forced input numerical comparison, not a broad task-quality guarantee','rows':[]}
with a.reference.open('rb') as f,a.candidate.open('rb') as g:
    h=header(f);assert h==header(g);v,n,p,raw=h;tokens=struct.unpack('i'*(p+n-1),raw)
    ref_nll=[];act_nll=[]
    for i in range(n):
        pos,t,x=row(f,v);pos2,u,y=row(g,v);assert pos==pos2
        mx,my=max(x),max(y);zx=mx+math.log(sum(math.exp(q-mx) for q in x));zy=my+math.log(sum(math.exp(q-my) for q in y))
        kl=sum(math.exp(q-zx)*(q-zx-w+zy) for q,w in zip(x,y))
        nmse=sum((q-w)**2 for q,w in zip(x,y))/max(sum(q*q for q in x),1e-30)
        result['rows'].append({'position':pos,'top1_same':t==u,'kl':max(kl,0),'nmse':nmse,'max_abs':max(abs(q-w) for q,w in zip(x,y))})
        if p+i<len(tokens):ref_nll.append(zx-x[tokens[p+i]]);act_nll.append(zy-y[tokens[p+i]])
    assert not f.read(1) and not g.read(1)
kls=sorted(x['kl'] for x in result['rows'])
result['summary']={'positions':n,'top1_matches':sum(x['top1_same'] for x in result['rows']),'mean_KL':sum(kls)/n,'median_KL':kls[n//2],'p95_KL':kls[min(n-1,int(n*.95))],'max_KL':max(kls),'PPL_ratio':math.exp(sum(act_nll)/len(act_nll)-sum(ref_nll)/len(ref_nll)), 'mean_reference_nll':sum(ref_nll)/len(ref_nll),'mean_candidate_nll':sum(act_nll)/len(act_nll)}
result['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.reference,a.candidate]}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']),flush=True)
