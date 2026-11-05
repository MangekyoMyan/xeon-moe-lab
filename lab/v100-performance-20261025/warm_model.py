#!/usr/bin/env python3
import sys,fcntl,json
from pathlib import Path
root=Path(__file__).resolve().parent;lab=root.parent;out=lab/'results'/root.name
sys.path.insert(0,str(lab/'board-study-20261024'));import hardware_run as runner
sys.path.insert(0,str(lab));from glm_q8_experiment import environment
runner.OUT=out
command=[sys.executable,str(lab/'without_thp.py'),'numactl','--physcpubind=0-23','--interleave=0,1',str(lab/'review-glm-q8-columns4-20261020/build/bin/llama-bench'),'-m',str(lab/'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf'),'-t','24','-p','128','-n','64','-r','1','-b','128','-ub','128','-ngl','0','-lm','mmap','-lzm','off','-dev','none','-nkvo','1','-nopo','1','--repack','0','--cpu-mask','0xffffff','--cpu-strict','1','--offline','--progress','-o','json']
with (lab/'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:runner.run('cold-model-warmup',command,timeout=3600,env=environment())
 finally:runner.fix_owner()
