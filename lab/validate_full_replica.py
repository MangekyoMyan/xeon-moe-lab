#!/usr/bin/env python3
import hashlib,json,os,re,struct,subprocess,time
from pathlib import Path
from full_replica_experiment import LAB,ROOT,OUT,MODEL,BUILD,ACTIVE,BASE_ENV
out=OUT/'validation';out.mkdir()
report={'logits':[],'responses':[]}
def save(): (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for label,build,source,mode in [('original',ACTIVE,LAB/'llama-private-prefetch',0),('disabled',BUILD,ROOT/'source',0),('single',BUILD,ROOT/'source',2),('full',BUILD,ROOT/'source',1)]:
 binary=out/(label+'-logits')
 cmd=['c++','-std=c++17','-O2','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),str(LAB/'deepseek-logits-probe.cpp'),'-L'+str(build/'bin'),'-Wl,-rpath,'+str(build/'bin'),'-lllama','-lggml-cpu','-lggml-base','-pthread','-o',str(binary)]
 subprocess.run(cmd,check=True)
 cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(binary),str(MODEL),str(out/(label+'.bin'))]
 print(json.dumps({'logits':label}),flush=True)
 with (out/(label+'.stdout')).open('w') as stdout,(out/(label+'.stderr')).open('w') as stderr:
  subprocess.run(cmd,stdout=stdout,stderr=stderr,env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode)),check=True,timeout=900)
 raw=(out/(label+'.bin')).read_bytes();vocab,count,prefix=struct.unpack_from('<iii',raw)
 assert vocab==262144 and count==34 and prefix==127
 assert len(raw)==12+160*4+count*(8+vocab*4)
 report['logits'].append({'label':label,'mode':mode,'sha256':hashlib.sha256(raw).hexdigest(),'logits_compared':vocab*count,'command':cmd})
 save()
report['logits_bit_identical']=len({x['sha256'] for x in report['logits']})==1
save();assert report['logits_bit_identical']
print(json.dumps({'all_logits_identical':True,'count':vocab*count}),flush=True)
# The existing buffer probe checks partial writes and reads from both sockets.
binary=out/'buffer-probe'
subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT/'source/ggml/include'),str(LAB/'numa-replica-probe.cpp'),'-L'+str(BUILD/'bin'),'-Wl,-rpath,'+str(BUILD/'bin'),'-lggml-cpu','-lggml-base','-pthread','-o',str(binary)],check=True)
for mode in (1,2):
 with (out/f'buffer-{mode}.log').open('w') as log:
  subprocess.run(['numactl','--physcpubind=0-23','--interleave=0,1',str(binary),str(out/f'buffer-{mode}.bin')],stdout=log,stderr=subprocess.STDOUT,env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode)),check=True,timeout=600)
report['buffer_modes_identical']=(out/'buffer-1.bin').read_bytes()==(out/'buffer-2.bin').read_bytes()
assert report['buffer_modes_identical'];save()
# Fixed Japanese response with matched prompts and deterministic sampling.
prompt='以下のメモを読んで、最後の質問に一行の日本語で答えてください。\n検証用コード: BLUE-731。\n質問: 検証用コード、日本の首都、17+25の答えを教えてください。説明は不要です。'
(out/'prompt.txt').write_text(prompt)
for i,label in enumerate('ABBA'):
 d=out/f'response-{i}-{label}';d.mkdir();mode=0 if label=='A' else 1
 cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(BUILD/'bin/llama-completion'),'-m',str(MODEL),'-t','24','-tb','24','-ngl','0','-c','4096','-b','512','-ub','512','--load-mode','mmap','--cpu-mask','0xffffff','--cpu-strict','1','--cpu-mask-batch','0xffffff','--cpu-strict-batch','1','--jinja','--offline','-f',str(out/'prompt.txt'),'--single-turn','--simple-io','--color','off','--no-display-prompt','--reasoning','off','--temp','0','--seed','42','-n','64','--perf']
 print(json.dumps({'response':str(d)}),flush=True)
 with (d/'stdout.txt').open('w') as stdout,(d/'stderr.txt').open('w') as stderr:
  subprocess.run(cmd,stdout=stdout,stderr=stderr,env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode)),check=True,timeout=600)
 log=(d/'stderr.txt').read_text();text=(d/'stdout.txt').read_text()
 timings={name:float(m.group(1)) for name in ['load','prompt eval','eval'] if (m:=re.search(r'common_perf_print:\s+'+name+r' time\s*=\s*([\d.]+) ms',log))}
 eval_n=int(re.search(r'common_perf_print:\s+eval time\s*=.*?/\s*(\d+) runs',log).group(1))
 report['responses'].append({'label':label,'mode':mode,'response':text,'timings_ms':timings,'eval_tokens':eval_n,'command':cmd});save()
report['responses_identical']=len({x['response'] for x in report['responses']})==1
assert report['responses_identical']; report['complete']=True;save()
print(json.dumps({'validation_complete':True}),flush=True)
