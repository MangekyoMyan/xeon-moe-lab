from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent
OUT=ROOT/'sparse-cache'
SOURCE=LAB/'deepseek41-bringup-20261101/source'
BASE=LAB/'deepseek41-bringup-20261101/build/bin'
cmd=['g++','-O2','-std=c++17','-I'+str(OUT),'-I'+str(SOURCE/'include'),'-I'+str(SOURCE/'ggml/include'),'-I'+str(SOURCE/'ggml/src'),str(OUT/'test-model.cpp'),
     '-L'+str(OUT/'runtime'),'-L'+str(BASE),'-Wl,-rpath,'+str(OUT/'runtime'),'-Wl,-rpath,'+str(LAB/'gpu-stream-20261103/runtime'),'-Wl,-rpath,'+str(BASE),
     '-lllama','-lggml-cpu','-lggml-base','-lggml','-o',str(OUT/'test-model')]
subprocess.run(cmd,check=True,timeout=120)
fixture=OUT/'fixture.gguf'
if not fixture.exists():subprocess.run([str(LAB/'iq-cache-next-20261103/make_fixture'),str(fixture)],check=True,timeout=30)
items=[('blk.0.ffn_gate_exps.weight','iq2_s',512,256),('blk.0.ffn_up_exps.weight','iq3_s',512,256),('blk.0.ffn_down_exps.weight','iq3_xxs',256,512)]
for label,selections in [('full',[[0,1,2,3]]*3),('partial',[[1,3],[0,2],[3]]),('zero',[[],[],[]])]:
    selected=sum(map(len,selections));size=sum(((k//256*m*144*len(ids)+4095)//4096*4096) for (_,_,k,m),ids in zip(items,selections))
    rows=[f'XEON_IQ_CACHE_PLAN_V2 3 {selected} {size}']
    for (name,typ,k,m),ids in zip(items,selections):rows.append(f'{name} {typ} {k} {m} 4 {len(ids)}'+''.join(f' {i}' for i in ids))
    (OUT/f'fixture-{label}.plan').write_text('\n'.join(rows)+'\n')
(OUT/'MODEL_TEST_BUILD.json').write_text(json.dumps(cmd,indent=2)+'\n')
print('Built tiny complete-model loader/plan test and three fixture plans.')
