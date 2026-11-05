#!/usr/bin/env python3
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
sys.path.insert(0, str(LAB / 'board-study-20261024'))
import hardware_run as runner
sys.path.insert(0, str(LAB))
from glm_q8_experiment import environment

runner.OUT = OUT
command = [sys.executable, str(LAB / 'without_thp.py'), 'numactl', '--physcpubind=0-23', '--interleave=0,1',
           str(LAB / 'review-glm-q8-columns4-20261020/build/bin/llama-bench'), '-m',
           str(LAB / 'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00001-of-00002.gguf'),
           '-t', '24', '-p', '128', '-n', '128', '-r', '2', '-b', '128', '-ub', '128', '-ngl', '0',
           '-lm', 'mmap', '-lzm', 'off', '-dev', 'none', '-nkvo', '1', '-nopo', '1', '--repack', '0',
           '--cpu-mask', '0xffffff', '--cpu-strict', '1', '--offline', '--progress', '-o', 'json']
try:
    subprocess.run([sys.executable, str(LAB / 'board-study-20261024/check_health.py'), '--output', str(OUT / 'health-before.json')], check=True)
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.run('cold-warmup', command, timeout=3600, env=environment())
    (OUT / 'cold-warmup-notice.json').write_text(json.dumps({'scope':'Warm model after reboot. Not an accepted performance comparison; preparation work may overlap.'})+'\n')
finally:
    runner.fix_owner()
