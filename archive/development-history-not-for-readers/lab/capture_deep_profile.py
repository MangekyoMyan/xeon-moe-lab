#!/usr/bin/env python3
"""Collect CPU symbols, core counters and system-wide IMC samples in a timed loop."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LAB = Path(__file__).resolve().parent
HELPER = Path('/home/codex/projects/returned-review/xeon-review-20261019/tools/capture_phase.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prefill', 'decode'))
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a new output directory; an existing capture is preserved.')
    model = LAB / 'models/teamblobfish--DeepSeek-V4-Flash-GGUF/IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf'
    command = [sys.executable, str(HELPER), '--bench', str(args.bench.resolve()),
               '--model', str(model), '--phase', args.phase, '--mode', 'record',
               '--cpus', '0-23', '--nodes', '0,1', '--seconds', '12', '--output', str(output)]
    counters = []
    logs = []
    report = {'command': command, 'counter_seconds': 8,
              'environment': {key: os.environ.get(key) for key in ('GGML_CPU_MOE_PAIR', 'GGML_CPU_NUMA_REPLICA', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')},
              'note': 'Diagnostic runs only. Core counters attach to the benchmark; IMC counters cover the whole otherwise-idle host. Windows start after the timed-loop marker and are approximate.'}
    helper = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for line in helper.stdout:
            print(line, end='', flush=True)
            if not line.startswith('Capturing a '):
                continue
            assert not counters, 'Only one timed sample is allowed'
            children = Path(f'/proc/{helper.pid}/task/{helper.pid}/children').read_text().split()
            bench_pids = [pid for pid in children if Path(f'/proc/{pid}/exe').resolve().name == 'llama-bench']
            if len(bench_pids) != 1:
                raise RuntimeError(f'Expected one benchmark child, found {bench_pids}')
            events = 'task-clock,cycles:u,instructions:u,cycle_activity.stalls_l2_pending:u,context-switches,cpu-migrations,minor-faults,major-faults,duration_time'
            specs = [
                ('core', ['-p', bench_pids[0], '-e', events]),
                ('imc', ['-a', '-e', 'uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/,duration_time']),
            ]
            report['bench_pid'] = int(bench_pids[0])
            report['counters'] = []
            for name, options in specs:
                cmd = ['perf', 'stat', '-x', ',', '--no-big-num', '-o', str(output / f'{name}.csv')] + options + ['--', 'sleep', '8']
                log = (output / f'{name}.stderr.log').open('w')
                logs.append(log)
                process = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
                counters.append(process)
                report['counters'].append({'name': name, 'command': cmd, 'started_monotonic': time.monotonic()})
        report['helper_exit_code'] = helper.wait(timeout=15)
        for process, entry in zip(counters, report.get('counters', [])):
            entry['exit_code'] = process.wait(timeout=15)
        if report['helper_exit_code'] or len(counters) != 2 or any(p.returncode for p in counters):
            raise RuntimeError('One or more diagnostic captures failed; inspect saved logs.')
    finally:
        for process in counters + [helper]:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        helper.stdout.close()
        for log in logs:
            log.close()
        if output.exists():
            (output / 'sidecar-counters.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
