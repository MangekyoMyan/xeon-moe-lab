import json,os,re,subprocess,time
from full_replica_experiment import *
out=OUT/'responses-long';out.mkdir()
prompt='次の質問に、日本語で300字程度、具体例を含めて答えてください。古い2CPUサーバーでLLMを使う場合、メモリ帯域とCPU間通信は応答速度にどう関係しますか？'
(out/'prompt.txt').write_text(prompt); records=[]
for i,label in enumerate('ABCCBA'):
 d=out/f'{i}-{label}';d.mkdir();mode={'A':0,'B':1,'C':2}[label]
 cmd=['numactl','--physcpubind=0-23','--interleave=0,1',str(BUILD/'bin/llama-completion'),'-m',str(MODEL),'-t','24','-tb','24','-ngl','0','-c','4096','-b','512','-ub','512','--load-mode','mmap','--cpu-mask','0xffffff','--cpu-strict','1','--cpu-mask-batch','0xffffff','--cpu-strict-batch','1','--jinja','--offline','-f',str(out/'prompt.txt'),'--single-turn','--simple-io','--color','off','--no-display-prompt','--reasoning','off','--temp','0','--seed','42','-n','128','--perf']
 if label=='C':cmd=['python3',str(LAB/'without_thp.py')]+cmd
 print(json.dumps({'response':str(d)}),flush=True);samples=[];start=time.monotonic()
 with (d/'stdout.txt').open('w') as stdout,(d/'stderr.txt').open('w') as stderr:
  p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode)),stdin=subprocess.DEVNULL)
  try:
   while p.poll() is None:
    s=snapshot(p.pid)
    if s:samples.append(s)
    if time.monotonic()-start>600:raise TimeoutError()
    time.sleep(1)
  finally:
   if p.poll() is None:p.terminate()
   try:rc=p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();rc=p.wait()
 assert rc==0
 log=(d/'stderr.txt').read_text();text=(d/'stdout.txt').read_text()
 timings={name:float(m.group(1)) for name in ['load','prompt eval','eval'] if (m:=re.search(r'common_perf_print:\s+'+name+r' time\s*=\s*([\d.]+) ms',log))}
 eval_n=int(re.search(r'common_perf_print:\s+eval time\s*=.*?/\s*(\d+) runs',log).group(1))
 records.append({'label':label,'mode':mode,'response':text,'timings_ms':timings,'eval_tokens':eval_n,'command':cmd,'samples':samples})
 (out/'comparison.json').write_text(json.dumps({'responses':records,'complete':len(records)==6,'responses_identical':len({r['response'] for r in records})==1},ensure_ascii=False,indent=2))
 print(json.dumps({'done_response':label,'tokens':eval_n,'timings_ms':timings}),flush=True)
assert len({r['response'] for r in records})==1
assert len({r['eval_tokens'] for r in records})==1
