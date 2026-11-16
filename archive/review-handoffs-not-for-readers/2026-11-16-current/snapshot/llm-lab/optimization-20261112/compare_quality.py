from pathlib import Path
import argparse,array,hashlib,json,math,sys,zlib
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
def read(folder):
 j=json.loads((folder/'RESULT.json').read_text());assert j['complete'] and j['byteorder']==sys.byteorder
 out={}
 for row in j['cases']:
  f=folder/row['probability_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['probability_sha256']
  v=array.array('d');v.frombytes(zlib.decompress(f.read_bytes()));assert len(v)==j['vocab'];out[row['name']]=(row,v)
 return j,out
ref,rv=read(a.reference);cand,cv=read(a.candidate)
assert ref['inputs_sha256']==cand['inputs_sha256'] and ref['vocab']==cand['vocab'] and rv.keys()==cv.keys()
rows=[];epsilon=1e-30
for name,(r,x) in rv.items():
 c,y=cv[name];sx=math.fsum(x);sy=math.fsum(y);x=[z/sx for z in x];y=[z/sy for z in y]
 row={'name':name,'tokens':r['prompt_tokens'],'same_argmax':r['argmax_token']==c['argmax_token'],
      'same_sampled_token':r['sampled_token']==c['sampled_token'],
      'total_variation':.5*math.fsum(abs(p-q) for p,q in zip(x,y)),
      'max_probability_delta':max(abs(p-q) for p,q in zip(x,y)),
      'KL_ref_to_candidate_floor1e30':math.fsum(p*math.log(max(p,epsilon)/max(q,epsilon)) for p,q in zip(x,y) if p>0)}
 rows.append(row);print(json.dumps(row))
result={'reference':str(a.reference),'candidate':str(a.candidate),'cases':rows,'probability_floor':epsilon,
 'scope':'Next-token distributions at six fixed synthetic prefixes; equality/similarity here does not prove general quality or later routing equality.'}
(R/'QUALITY_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
