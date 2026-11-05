#!/usr/bin/env python3
"""Read-only diagnostics of the frozen active model; no runtime tuning/adoption."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

LAB = Path(__file__).resolve().parent
HELPER = LAB / 'observe_phase.py'
COMMON = ['task-clock', 'cycles:u', 'instructions:u', 'context-switches',
          'cpu-migrations', 'minor-faults', 'major-faults', 'duration_time']
EVENTS = {
    'stalls': ['cycle_activity.stalls_total:u', 'cycle_activity.stalls_l2_pending:u',
               'cycle_activity.cycles_l2_pending:u'],
    'locality': ['mem_load_uops_llc_miss_retired.local_dram:u',
                 'mem_load_uops_llc_miss_retired.remote_dram:u'],
    'outstanding': ['offcore_requests_outstanding.demand_data_rd:u',
                    'offcore_requests_outstanding.cycles_with_demand_data_rd:u',
                    'offcore_requests.demand_data_rd:u'],
}


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def snapshot(pid):
    data = {'monotonic': time.monotonic(), 'threads': {}, 'cpu_temperatures_mC': {},
            'throttle_counts': {}, 'proc_stat': read('/proc/stat'), 'vmstat': read('/proc/vmstat'),
            'io': read(f'/proc/{pid}/io'), 'status': read(f'/proc/{pid}/status')}
    for task in Path(f'/proc/{pid}/task').glob('*'):
        stat = read(task / 'stat')
        status = read(task / 'status')
        if stat is None or status is None:
            continue
        fields = stat.rsplit(')', 1)[1].split()
        keep = ('Cpus_allowed_list', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches')
        data['threads'][task.name] = {
            'utime_ticks': int(fields[11]), 'stime_ticks': int(fields[12]),
            'processor': int(fields[36]), 'state': fields[0],
            'schedstat': read(task / 'schedstat'),
            'status': {k: v.strip() for line in status.splitlines() if ':' in line
                       for k, v in [line.split(':', 1)] if k in keep},
        }
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        if (read(hw / 'name') or '').strip() != 'coretemp':
            continue
        for p in hw.glob('temp*_input'):
            data['cpu_temperatures_mC'][hw.name + '/' + (read(p.with_name(p.name.replace('_input', '_label'))) or p.name).strip()] = int(p.read_text())
    for p in Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*throttle_count'):
        data['throttle_counts'][str(p)] = int(p.read_text())
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['decode', 'prefill'])
    ap.add_argument('events', choices=EVENTS)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bench', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--depth', type=int, default=0)
    ap.add_argument('--window', type=int, default=18)
    ap.add_argument('--trace-futex', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.window <= 56:
        ap.error('Window must be between 1 and 56 seconds')
    out = args.output.resolve()
    if out.exists():
        ap.error('Output must be new')
    bench = args.bench.resolve()
    model = args.model.resolve()
    cmd = [sys.executable, str(HELPER), '--bench', str(bench), '--model', str(model),
           '--threads', str(args.threads), '--tokens', str(args.tokens), '--phase', args.phase, '--mode', 'futex' if args.trace_futex else 'wait', '--cpus', '0-23', '--nodes', '0,1',
           '--seconds', str(args.window + 2), '--depth', str(args.depth), '--output', str(out)]
    report = {'command': cmd, 'events': args.events, 'counter_seconds': args.window,
              'note': 'Approximate timed-loop window, warm cache. System-wide IMC counters include other host activity. Performance includes observer overhead. Stalls, sampled symbols, and DRAM traffic overlap; they are not additive time fractions.',
              'clk_tck': os.sysconf('SC_CLK_TCK'), 'counters': [], 'telemetry': [],
              'environment': {k: v for k, v in os.environ.items() if k.startswith(('OMP_', 'GOMP_', 'GGML_CPU_'))}}
    helper = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    procs, logs, thread = [], [], None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    def collect(pid):
        try:
            (out / 'numa-start.txt').write_text(Path(f'/proc/{pid}/numa_maps').read_text())
            start = time.monotonic()
            for i in range(args.window + 1):
                if i:
                    time.sleep(max(0, start + i - time.monotonic()))
                report['telemetry'].append(snapshot(pid))
            (out / 'numa-end.txt').write_text(Path(f'/proc/{pid}/numa_maps').read_text())
        except Exception as exc:
            report['telemetry_error'] = repr(exc)
    try:
        for line in helper.stdout:
            print(line, end='', flush=True)
            if not line.startswith('Capturing a '):
                continue
            if procs:
                raise RuntimeError('Unexpected second window')
            children = Path(f'/proc/{helper.pid}/task/{helper.pid}/children').read_text().split()
            pids = [p for p in children if Path(f'/proc/{p}/exe').resolve().name == 'llama-bench']
            if len(pids) != 1:
                raise RuntimeError(f'Ambiguous benchmark child: {pids}')
            pid = pids[0]
            report['bench_pid'] = int(pid)
            common = [e for e in COMMON if args.events != 'locality' or e not in ('cycles:u', 'instructions:u')]
            specs = [('core', ['-p', pid, '-e', ','.join(common + EVENTS[args.events])]),
                     ('imc', ['-a', '-A', '-I', '1000', '-e', 'uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/'])]
            for name, options in specs:
                command = ['perf', 'stat', '-x', ',', '--no-big-num', '-o', str(out / f'{name}.csv')] + options + ['--', 'sleep', str(args.window)]
                log = (out / f'{name}.stderr.log').open('w'); logs.append(log)
                report['counters'].append({'name': name, 'command': command, 'start_monotonic': time.monotonic()})
                procs.append(subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True))
            thread = threading.Thread(target=collect, args=(pid,)); thread.start()
        report['helper_exit_code'] = helper.wait(timeout=10)
        for process, entry in zip(procs, report['counters']):
            entry['exit_code'] = process.wait(timeout=10)
        if thread:
            thread.join(timeout=25)
        if report['helper_exit_code'] or len(procs) != 2 or any(p.returncode for p in procs) or report.get('telemetry_error'):
            raise RuntimeError('Capture failed; inspect evidence')
        cap = json.loads((out / 'capture.json').read_text())
        import csv
        rows = [row for row in csv.reader((out / 'core.csv').read_text().splitlines()) if row and not row[0].startswith('#')]
        report['counters_unmultiplexed'] = all(float(row[-3]) >= 99.9 and '<not' not in row[0] for row in rows)
        excluded = [','.join(row[2:-4]) for row in rows if float(row[-3]) < 99.9 or '<not' in row[0]]
        report['excluded_supplemental_events'] = excluded
        report['primary_counters_unmultiplexed'] = not any(event != 'cycle_activity.cycles_l2_pending:u' for event in excluded)
        report['valid_approximate_sample'] = cap['valid_sample'] and len(report['telemetry']) == args.window + 1 and report['primary_counters_unmultiplexed']
        if not report['valid_approximate_sample']:
            raise RuntimeError('Counter multiplexing or incomplete observation; exclude this sample')
    finally:
        for process in procs + [helper]:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
        if thread:
            thread.join(timeout=25)
        helper.stdout.close()
        for log in logs:
            log.close()
        if out.exists():
            (out / 'observations.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
