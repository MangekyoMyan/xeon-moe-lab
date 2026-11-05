#!/usr/bin/env python3
from pathlib import Path
import json
import os
import subprocess

R=Path(__file__).resolve().parent;LAB=R.parent
out=R/'extra-validation';out.mkdir(exist_ok=False)
commands=[]
def run(cmd,**kwargs):
    commands.append(cmd)
    return subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=90,**kwargs)

p=run(['python3',str(R/'run_micro.py'),'--variant','sequential4','--portable'])
(out/'portable.log').write_text(p.stdout+p.stderr)
source=(LAB/'pro-review-received-20261102/deepseek-cpu-review-result/route_test/test.cpp').read_text()
source=source.replace('for (int threads : {1, 3})','for (int threads : {1, 3, 12, 24})')
source=source.replace('for (int rows : {3, 17})','for (int rows : {3, 4, 5, 6, 7, 15, 16, 17, 18, 19})')
source=source.replace('t + 1 == tokens ? -1 : (duplicate ? 0 : 1)',
                      '(t + 1 == tokens && (!duplicate || target_rows % 2)) ? -1 : (duplicate ? 0 : 1)')
source=source.replace('for (int t = 0; t + 1 < tokens; t++) {\n            duplicate_equal',
                     'for (int t = 0; t < tokens; t++) {\n            if (routes[t * used + 1] < 0) continue;\n            duplicate_equal')
(out/'route.cpp').write_text(source)
lib=R/'hybrid/build-cpu-ivybridge/bin'
run(['g++','-std=c++17','-O3','-march=ivybridge','-mtune=ivybridge','-mno-avx2','-mno-fma',
     '-I'+str(R/'hybrid/engine/ggml/include'),str(out/'route.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
     '-lggml-cpu','-lggml-base','-o',str(out/'route')])
p=run([str(out/'route')]);(out/'route.jsonl').write_text(p.stdout);(out/'route.stderr').write_text(p.stderr)
route=[json.loads(s) for s in p.stdout.splitlines()];assert route[-1]['cases']==640 and route[-1]['failures']==0

probe=LAB/'mxfp4-target-validation-20261102/stock/build-cpu-ivybridge/bin/cpu-quant-probe'
args=['--type','all','--mode','both','--k','512','--m','259','--experts','4','--topk','2','--routing','shared',
      '--tokens','1,2,3,4,5,6,7,8,17,18,19','--threads','12','--strict-cpu','--input-pad','7','--warmups','0','--repeats','1']
common=['numactl','--interleave=0,1','taskset','-c','0-11',str(probe),*args]
baseline=run([*common,'--dump-prefix',str(out/'stock')]);(out/'profile-baseline.jsonl').write_text(baseline.stdout)
env=dict(os.environ,LD_LIBRARY_PATH=str(R/'profile-stock')+':'+str(LAB/'deepseek41-bringup-20261101/build/bin'),
         GGML_MOE_HIST_FILE=str(out/'histogram.jsonl'))
profile=run([*common,'--dump-prefix',str(out/'profile')],env=env);(out/'profile-result.jsonl').write_text(profile.stdout)
a=[json.loads(s) for s in baseline.stdout.splitlines()];b=[json.loads(s) for s in profile.stdout.splitlines()]
assert a[-1]['failures']==b[-1]['failures']==0 and a[-1]['cases']==b[-1]['cases']==88
for x,y in zip(a[:-1],b[:-1]):
    for key in ['type','k','m','tokens','weight_fnv1a64','input_fnv1a64','route_fnv1a64','output_fnv1a64']:assert x[key]==y[key]
dump_count=0
for p in out.glob('stock-*.f32'):
    assert p.read_bytes()==(out/p.name.replace('stock-','profile-',1)).read_bytes();dump_count+=1
assert dump_count==88
hist=[json.loads(s) for s in (out/'histogram.jsonl').read_text().splitlines()];assert len(hist)==88
for h in hist:
    bins=h['hist_0_1_2_3_4to7_8to15_16to31_32plus'];n=h['batch_tokens']
    idx=n if n<4 else 4 if n<8 else 5 if n<16 else 6 if n<32 else 7
    expected=[0]*8;expected[0]=2;expected[idx]=2
    assert bins==expected and sum(bins)==4 and h['active_experts']==2 and h['assigned_rows']==n*2
    if h['mx421_plan_eligible']:assert sum(h['planned_rows_4_2_1'])==h['assigned_rows']
chunk_checks=[]
for threads,mask,want in [(1,'0',[32,0,4]),(12,'0-11',[32,4,0])]:
    path=out/f'chunk-t{threads}.jsonl';ev=dict(env,GGML_MOE_HIST_FILE=str(path))
    run(['taskset','-c',mask,str(probe),'--type','mxfp4','--mode','up','--k','512','--m','33','--tokens','18',
         '--experts','2','--topk','2','--routing','shared','--threads',str(threads),'--strict-cpu','--warmups','0','--repeats','1'],env=ev)
    h=json.loads(path.read_text());assert h['planned_rows_4_2_1']==want
    chunk_checks.append({'threads':threads,'tokens_per_expert':18,'planned_rows_4_2_1':want})
record={'portable_passed':True,'route_cases':640,'route_outputs':sum(x['outputs'] for x in route[:-1]),'route_bit_mismatches':sum(x['bit_mismatches'] for x in route[:-1]),
        'profile_cases':88,'profile_full_output_bit_equal':True,'histogram_zeros_and_counts_checked':True,'chunk_boundary_checks':chunk_checks,'commands':commands}
(out/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='commands'}),flush=True)
