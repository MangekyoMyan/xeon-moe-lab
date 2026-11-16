"""GPU operator equivalence only. Requires the API already stopped; never changes services."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent.parent;LAB=R.parent;E=LAB/'deepseek41-bringup-20261101'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode==0:raise RuntimeError('Stop the API explicitly before isolated GPU tests')
 raw=(E/'server-config.json').read_bytes();config=json.loads(raw)
 cases=json.loads((LAB/'expert-validation-20261101/cases.json').read_text());assert len(cases)==8
 build=json.loads((R/'cuda-sort/BUILD.json').read_text());variants={'stock':{'library':str(E/'build/bin/libggml-cuda.so.0.28.0'),'sha256':digest(E/'build/bin/libggml-cuda.so.0.28.0')},**build['variants']}
 folder=R/'cuda-sort'/('gpu-'+time.strftime('%Y%m%d-%H%M%S'));folder.mkdir(mode=0o700)
 env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_')) and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES']}
 env.update(OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='24',OMP_MAX_ACTIVE_LEVELS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=config['environment']['CUDA_VISIBLE_DEVICES'])
 cpu=R/'sparse-cache/release-v2-a';base=LAB/'gpu-stream-20261103/runtime';cuda=LAB/'v100-prep-20261024/cuda-12.9.2/lib'
 record={'config_sha256':hashlib.sha256(raw).hexdigest(),'probe_sha256':digest(R/'cuda-sort/test-gpu'),'cuda_variants':variants,'cases':[],'complete':False,'error':None}
 def save():(folder/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
 save()
 try:
  for case in cases:
   values={}
   for label,item in variants.items():
    assert (E/'server-config.json').read_bytes()==raw
    assert subprocess.run(['systemctl','is-active','--quiet','llm-deepseek41-api.service']).returncode!=0
    library=Path(item['library']);assert digest(library)==item['sha256']
    env['LD_LIBRARY_PATH']=':'.join(map(str,[library.parent,cpu,base,E/'build/bin',cuda]))
    result=folder/(case['name']+'-'+label+'.jsonl');command=['numactl','--physcpubind=6','--membind=1',str(R/'cuda-sort/test-gpu'),case['shard'],case['tensor'],str(result),str(library)]
    start=time.monotonic();p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
    (folder/(case['name']+'-'+label+'.stdout')).write_text(p.stdout);(folder/(case['name']+'-'+label+'.stderr')).write_text(p.stderr);p.check_returncode()
    values[label]=[json.loads(x) for x in result.read_text().splitlines()];assert len(values[label])==11
   assert values['stock']==values['control']==values['candidate'],'CUDA output bytes hash differs: '+case['name']
   row={'name':case['name'],'cases':11,'outputs_each':sum(x['outputs'] for x in values['stock']),'all_output_byte_hashes_equal':True,'repeat_memcmp_passed':True,'fallback_cases':sum(x['fallback_verified'] for x in values['stock'])};record['cases'].append(row);save();print(json.dumps(row),flush=True)
  record['complete']=True
 except BaseException as error:record['error']=repr(error);raise
 finally:save()
 print('Saved '+str(folder),flush=True)
if __name__=='__main__':main()
