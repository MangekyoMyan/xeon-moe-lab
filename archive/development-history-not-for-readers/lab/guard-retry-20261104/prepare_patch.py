from pathlib import Path
import hashlib,difflib
R=Path(__file__).resolve().parent;src=R/'supervise.before.py';old=src.read_text()
if (R/'supervise.py').exists():raise SystemExit('Candidate already prepared; preserve subsequent reviewed edits.')
assert hashlib.sha256(src.read_bytes()).hexdigest()=='f31ec8ae7176a549e6ebc2494f3a23e77f92c3bd8d0dca6d8ea4a4512bfdafb7'
s=old
def replace(a,b):
 global s
 assert s.count(a)==1,a[:80];s=s.replace(a,b)
replace('import time\n','import time\n\nfrom gpu_guard import GpuProbe, ProbeCancelled, temperatures\n')
replace('\ndef main():','''
def check_host(initial_host):
    cpu = watch.temperatures()
    temperatures(cpu, 60, 'CPU')
    if not initial_host or watch.counters() != initial_host:
        raise RuntimeError('Host ECC/throttle counter changed or missing')
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                         if l.startswith('MemAvailable:')))
    if available < 3 * 1024**2:
        raise RuntimeError('Host available memory below3GiB')
    return cpu, available


def main():''')
replace('    initial_host = watch.counters()','    initial_host = watch.counters()\n    check_host(initial_host)')
replace('                  status=\'starting\', initial_gpu=initial, initial_fans=fans, latest_bmc=bmc,', "                  status='starting', initial_gpu=initial, initial_host=initial_host, initial_fans=fans, latest_bmc=bmc,")
replace('    exit_code = 0\n    try:', '''    exit_code = 0

    def probe_event(event):
        record['gpu_monitor'] = dict(probe.stats)
        record['updated_unix'] = time.time()
        write_json(ROOT / 'health.json', record)
        print('GPU monitoring: ' + json.dumps(event), file=sys.stderr, flush=True)

    probe = GpuProbe(watch.gpu_sample, initial, initial_host, record['boot_id'],
                     check_host=lambda: check_host(initial_host),
                     cancelled=lambda: stop_requested or (child is not None and child.poll() is not None),
                     emit=probe_event)
    record['gpu_monitor'] = dict(probe.stats)
    try:''')
a=s.index('                gpu = watch.gpu_sample()');b=s.index('                if now >= next_fans:',a)
s=s[:a]+'''                gpu = probe.read()
                cpu, available = check_host(initial_host)
'''+s[b:]
replace('    except BaseException as error:', '''    except ProbeCancelled:
        if stop_requested:
            record['status'] = 'stopped_by_request'
        else:
            record['status'] = 'exited'
            exit_code = child.returncode or 0
    except BaseException as error:''')
replace("        record['finished_unix'] = time.time()", "        record['finished_unix'] = time.time()\n        record['gpu_monitor'] = dict(probe.stats)")
(R/'supervise.py').write_text(s)
(R/'supervise.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='supervise.before.py',tofile='supervise.py')))
