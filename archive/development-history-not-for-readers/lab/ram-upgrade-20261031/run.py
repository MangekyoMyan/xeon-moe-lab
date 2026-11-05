#!/usr/bin/env python3
"""Repeat preserved CPU-only benchmarks after RAM upgrade; no models/config writes."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
OUT = LAB / 'results/ram-upgrade-20261031'
sys.path[:0] = [str(HERE), str(LAB), str(LAB / 'board-study-20261024')]
import memory_runner as mem
import quant_runner as quant
import hardware_run as hw

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def capture(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def inventory(label):
    data = {'unix': time.time(), 'sample': mem.sample(), 'kernel': capture(['uname','-a']),
            'dimms': capture(['dmidecode','--type','17']), 'numa': capture(['numactl','--hardware']),
            'cpu': capture(['lscpu']), 'gpu': capture(['nvidia-smi','-q','-x']),
            'api': capture(['systemctl','is-active','llm-openai-api.service']),
            'processes': capture(['ps','-eo','pid,comm,pcpu,pmem','--sort=-pcpu']),
            'meminfo': Path('/proc/meminfo').read_text(),
            'bmc_sensors': hw.ipmi('sensor','list'), 'bmc_sel': hw.ipmi('sel','elist'),
            'fan_mode': hw.ipmi('raw','0x30','0x45','0x00'),
            'fan_duty_raw': [hw.ipmi('raw','0x30','0x90','0x5a','0x03',hex(r),'0x01') for r in range(0x10,0x14)],
            'counters': hw.counters(),
            'cpufreq': {str(p): p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/*')
                        if p.name in ['scaling_governor','scaling_min_freq','scaling_max_freq','scaling_driver']}}
    write(OUT / f'{label}.json', data)
    return data

def verify_binaries():
    expected = json.loads((LAB / 'quantize-tradeoff-20261021/source-and-binary-hashes.json').read_text())
    results = {name: {'expected': sha, 'actual': hashlib.sha256(Path(name).read_bytes()).hexdigest()}
               for name, sha in expected.items()}
    assert all(v['expected'] == v['actual'] for v in results.values())
    bins = [LAB / 'server-study-20261023/memory-probe', quant.BIN / 'quantize-matvec', quant.BIN / 'test-quantize-perf-fixed']
    libraries = {}
    for binary in bins:
        linked = capture(['ldd', str(binary)])
        libraries[str(binary)] = linked
        for line in linked['stdout'].splitlines():
            if '=>' in line and 'ggml' in line:
                path = Path(line.split('=>', 1)[1].strip().split()[0])
                results[str(path)] = {'actual': hashlib.sha256(path.read_bytes()).hexdigest()}
    for binary in bins:
        results.setdefault(str(binary), {})['actual'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    write(OUT / 'binary-verification.json', {'hashes': results, 'ldd': libraries})

def check_safety(initial):
    assert hw.counters() == initial, 'ECC/thermal throttle counter changed'
    temps = hw.temperatures()
    assert temps and max(temps.values()) < 60, 'CPU temperature cutoff'
    return temps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['inventory','memory','quant','counters','local','after'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase in ('inventory','after'):
            data = inventory('before' if args.phase == 'inventory' else 'after')
            if args.phase == 'inventory':
                verify_binaries()
                assert data['api']['stdout'].strip() == 'inactive'
                assert all(int(v.strip(),16) <= 178 for v in data['fan_duty_raw'])
            print(json.dumps({'phase': args.phase, 'CPU_C': data['sample']['temperature_C'],
                              'ECC': data['sample']['ECC'], 'fan_mode': data['fan_mode'],
                              'fan_duty_raw': data['fan_duty_raw']}), flush=True)
            return
        initial = hw.counters()
        assert all(v == 0 for k,v in initial.items() if 'edac' in k), 'Investigate new-boot ECC first'
        check_safety(initial)
        quant.temperatures = lambda: check_safety(initial)
        if args.phase == 'memory':
            mem.OUT.mkdir(parents=True, exist_ok=True)
            cases = [
                ('read_n0_local','read',6,'node0',0,0,1024),
                ('read_n1_local','read',6,'node1',1,1,1024),
                ('read_both_physical','read',12,'both',-1,-1,1024),
                ('read_both_SMT','read',24,'both',-1,-1,1024),
                ('read_n0_SMT','read',12,'node0',0,0,1024),
                ('read_n1_SMT','read',12,'node1',1,1,1024),
                ('read_n0_remote','read',6,'node0',1,1,1024),
                ('read_n1_remote','read',6,'node1',0,0,1024),
                ('latency_n0_local','latency',1,'node0',0,0,128),
                ('latency_n0_remote','latency',1,'node0',1,1,128)]
            for r in range(3):
                for name,*case in (cases if r % 2 == 0 else cases[::-1]):
                    mem.run(f'{name}_r{r}',*case,4)
                    check_safety(initial)
        elif args.phase == 'quant':
            quant.OUT.mkdir(parents=True, exist_ok=True)
            one = ['numactl','--physcpubind=0','--membind=0']
            both = ['numactl','--physcpubind=0-23','--interleave=0,1']
            types = [arg for kind in quant.TYPES for arg in ['--type',kind]]
            for r in range(2):
                for regime,size,count in [('cache-long','4096','500000'),('stream','33554432','20')]:
                    quant.run(f'quant-perf-{regime}-{r}', one + [str(quant.BIN/'test-quantize-perf-fixed'),
                              *types,'--op','vec_dot_q','--op','dequantize_row_q','--size',size,'-i',count],timeout=300)
            for threads in (1,12,24):
                for regime,rows in [('cache',24),('stream',65536)]:
                    quant.run(f'matvec-k4096-{regime}-t{threads}', both + [str(quant.BIN/'quantize-matvec'),
                              '4096',str(rows),str(threads),'0.25','6'],timeout=180)
        elif args.phase == 'local':
            import statistics
            quant.OUT = OUT / 'local'
            quant.OUT.mkdir(parents=True, exist_ok=True)
            baseline = json.loads((LAB/'results/bottleneck-observation-20261019/streaming-reference.json').read_text())
            assert hashlib.sha256((LAB/'read-bandwidth').read_bytes()).hexdigest() == baseline['sha256']['read-bandwidth']
            default_env = quant.environment
            def local_env():
                env = default_env()
                env.update(OMP_PROC_BIND='true',OMP_PLACES='threads',OMP_DYNAMIC='FALSE',OMP_THREAD_LIMIT='12')
                return env
            quant.environment = local_env
            for i,placement in enumerate(('local','interleave','interleave','local','local','interleave')):
                name = f'read-{placement}-{i}'
                quant.run(name, ['numactl','--physcpubind=0-11','--localalloc' if placement == 'local' else '--interleave=0,1',
                               str(LAB/'read-bandwidth'),'12'], timeout=30)
                data = json.loads((quant.OUT/name/'stdout.txt').read_text())
                assert sorted(data['cpus']) == list(range(12)) and data['checksum_passed']
                print(name, statistics.median(data['gb_s']), flush=True)
            quant.environment = default_env
            for threads in (12,24):
                quant.run(f'matvec-local-t{threads}', ['numactl','--physcpubind=0-23','--localalloc',
                          str(quant.BIN/'quantize-matvec'),'4096','65536',str(threads),'0.25','6'], timeout=180)
        elif args.phase == 'counters':
            import observe_quantize_tradeoff as obs
            obs.OUT = OUT / 'counters'
            obs.OUT.mkdir(parents=True, exist_ok=True)
            old_snapshot = obs.snapshot
            def safe_snapshot(pid):
                try:
                    check_safety(initial)
                except BaseException:
                    os.killpg(pid, signal.SIGTERM)
                    raise
                return old_snapshot(pid)
            obs.snapshot = safe_snapshot
            for kind in ('q8_0','q6_K','q4_K','q3_K'):
                obs.capture(kind)
                check_safety(initial)
        write(OUT / f'{args.phase}-complete.json', {'complete': True, 'unix': time.time(), 'after': mem.sample()})
    owner = LAB.stat()
    for p in [OUT, *OUT.rglob('*')]:
        os.chown(p, owner.st_uid, owner.st_gid, follow_symlinks=False)

if __name__ == '__main__':
    main()
