#!/usr/bin/env python3
"""Collect sampled load use-latency; excludes all simultaneous core counters."""
import json
from pathlib import Path
import subprocess
import sys

lab = Path(__file__).resolve().parent
root = lab / 'results/bottleneck-observation-20261019'
model = lab / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf'
records = []
for phase in ('decode', 'prefill'):
    output = root / (phase + '-memory-1')
    cmd = ['sudo', '-n', 'env', 'OMP_DYNAMIC=FALSE', 'OMP_THREAD_LIMIT=24',
           'OMP_MAX_ACTIVE_LEVELS=1', 'GGML_CPU_NUMA_REPLICA=0', 'GGML_CPU_MOE_PAIR_TRACE=0',
           'GGML_CPU_NUMA_REPLICA_TRACE=0', sys.executable, str(lab / 'observe_phase.py'),
           '--bench', str(lab / 'build-avx-deep/bin/llama-bench'), '--model', str(model),
           '--phase', phase, '--mode', 'memory', '--cpus', '0-23', '--nodes', '0,1',
           '--seconds', '20', '--output', str(output)]
    print('Starting memory sample ' + phase, flush=True)
    with (root / (phase + '-memory-1.log')).open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    report_cmd = ['sudo', '-n', 'perf', 'report', '--stdio', '--mem-mode', '--no-children', '-g', 'none',
                  '--sort', 'mem,symbol', '--show-nr-samples', '--show-total-period',
                  '--percent-limit', '0', '-t', ';', '-i', str(output / 'profile.data')]
    with (root / (phase + '-memory-symbols.csv')).open('w') as f:
        subprocess.run(report_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    script_cmd = ['sudo', '-n', 'perf', 'script', '-i', str(output / 'profile.data'),
                  '-F', 'tid,period,weight,data_src,sym']
    with (root / (phase + '-memory-samples.txt')).open('w') as f:
        subprocess.run(script_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    records.append({'phase': phase, 'capture_command': cmd, 'report_command': report_cmd, 'script_command': script_cmd})
    (root / 'memory-index.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Finished memory sample ' + phase, flush=True)

output = root / 'prefill-full-stalls-futex-1'
cmd = ['sudo', '-n', 'env', 'OMP_DYNAMIC=FALSE', 'OMP_THREAD_LIMIT=24',
       'OMP_MAX_ACTIVE_LEVELS=1', 'GGML_CPU_NUMA_REPLICA=0', 'GGML_CPU_MOE_PAIR_TRACE=0',
       'GGML_CPU_NUMA_REPLICA_TRACE=0', sys.executable, str(lab / 'observe_bottlenecks.py'),
       'prefill', 'stalls', '--window', '56', '--trace-futex', '--output', str(output)]
print('Starting prefill futex observation', flush=True)
with (root / 'prefill-futex-1.log').open('w') as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
with (root / 'prefill-futex-events.txt').open('w') as f:
    subprocess.run(['sudo', '-n', 'perf', 'script', '--ns', '-i', str(output / 'profile.data'),
                    '-F', 'tid,time,event,trace'], stdout=f, stderr=subprocess.STDOUT, check=True)
print('Finished prefill futex observation', flush=True)
