#!/usr/bin/env python3
"""Sequential monitored experiments. No source edits/builds during timed runs."""
import json,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent
while not json.loads((R/'preload-state.json').read_text()).get('completed'):
    print('Waiting for HDD preload before performance measurements',flush=True);time.sleep(30)
# Wait until watcher finishes final counters and releases the shared lock.
import fcntl
with (R.parent/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)

def run(label,opts,power=150):
    assert not (R/'STOP_REQUESTED').exists()
    cmd=['python3',str(R/'watch.py'),'--label',label,'--power-limit',str(power),'--timeout','900','--',
         'python3',str(R/'run_probe.py'),label,*opts]
    print('Starting',label,flush=True)
    subprocess.run(cmd,check=True)
    result=json.loads((R/'model-runs'/label/'result.json').read_text())
    print('Result',label,result['XV_TIMING'],flush=True)
    with (R/'JOURNAL.jsonl').open('a') as f:
        f.write(json.dumps({'event':'model_run_completed','label':label,'power_W':power,'result':result})+'\n')

run('e00-original-parity',['--original','--records','18','--text','parity-text.txt','--save-logits'])
run('e00-overlay-parity',['--records','18','--text','parity-text.txt','--save-logits'])
a=(R/'model-runs/e00-original-parity/logits.bin').read_bytes()
b=(R/'model-runs/e00-overlay-parity/logits.bin').read_bytes()
assert a==b, 'Disabled experimental overlay differs from original engine'
run('e01-rows-parity',['--rows','1','--records','18','--text','parity-text.txt','--save-logits'])
assert a==(R/'model-runs/e01-rows-parity/logits.bin').read_bytes(),'Rows4 model output differs'
for index,rows in enumerate(['0','1','4096','1408','1408','4096','1','0']):
    run('e01-%02d-rows%s'%(index,rows),['--rows',rows,'--records','192','--measure'])
run('e02-axis-quality',['--axis','1','--records','256','--save-logits','--measure'])
run('e00-split-quality',['--records','256','--save-logits','--measure'])
run('e03-mlp-quality',['--mlp','--records','256','--save-logits','--measure'])
print('Stage1 completed. Review metrics/numerics before selecting next experiments.',flush=True)
