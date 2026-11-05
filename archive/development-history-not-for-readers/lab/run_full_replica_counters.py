import json,os,subprocess,sys
from full_replica_experiment import *
records=[]
for i,label in enumerate('ABC'):
 mode=1 if label=='B' else 2
 d=OUT/f'locality-{i}-{label}'
 cmd=[sys.executable,str(LAB/'observe_full_replica.py'),'decode','locality','--bench',str(BUILD/'bin/llama-bench'),'--model',str(MODEL),'--threads','24','--tokens','512','--window','18','--output',str(d)]
 if label=='C':cmd=[sys.executable,str(LAB/'without_thp.py')]+cmd
 print(json.dumps({'counter_start':label,'mode':mode}),flush=True)
 with (OUT/f'locality-{i}-{label}.log').open('w') as log:
  subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(BASE_ENV,GGML_CPU_NUMA_FULL=str(mode)),check=True,timeout=600)
 records.append({'label':label,'mode':mode,'directory':str(d)})
 (OUT/'locality-index.json').write_text(json.dumps(records,indent=2))
 print(json.dumps({'counter_done':label}),flush=True)
