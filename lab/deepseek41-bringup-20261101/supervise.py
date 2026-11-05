#!/usr/bin/python3
"""Run the private API with bounded health records and a thermal stop guard."""
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gpu_guard import GpuProbe, ProbeCancelled, temperatures

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
spec = importlib.util.spec_from_file_location('hybrid_watch', LAB / 'moe-engine-20261029/watch.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_bmc(value):
    if 'ECC' in value['sel']:
        raise RuntimeError('BMC reports an ECC event')
    for line in value['sensors'].splitlines():
        fields = [x.strip() for x in line.split('|')]
        if len(fields) < 2 or fields[1] == 'na':
            continue
        if fields[0] == 'Peripheral Temp' and float(fields[1]) >= 45:
            raise RuntimeError('Peripheral temperature limit')
        if fields[0] == 'PCH Temp' and float(fields[1]) >= 55:
            raise RuntimeError('PCH temperature limit')


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


def main():
    assert os.geteuid() == 0
    config = json.loads((ROOT / 'server-config.json').read_text())
    if (ROOT / 'STOP_REQUESTED').exists() or (LAB / 'moe-engine-20261029/STOP_REQUESTED').exists():
        raise RuntimeError('Inspect the previous hardware stop before starting')
    lock = (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = watch.gpu_sample()
    if abs(initial['limit_W'] - 150) >= .5:
        raise RuntimeError('GPU must already be capped at150W')
    if initial['ecc'] != 'Enabled' or initial['ecc_double'] != 0:
        raise RuntimeError('GPU ECC preflight failed')
    if initial['gpu_C'] >= 55 or initial['hbm_C'] >= 55:
        raise RuntimeError('GPU too warm to start')
    fans = watch.fan_duty_sample()
    if fans['mode'] != '00' or max(fans['raw']) > 178:
        raise RuntimeError('Fan policy must be Standard and <=70percent')
    initial_host = watch.counters()
    check_host(initial_host)
    bmc = watch.bmc_sample()
    check_bmc(bmc)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    if busy:
        raise RuntimeError('GPU already in use')
    owner = LAB.stat()
    env = watch.environment()
    for key in list(env):
        if key.startswith(('LLAMA_', 'GGML_', 'OMP_', 'GOMP_')) or key in ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES']:
            env.pop(key)
    env.update(config['environment'])
    record = dict(started_unix=time.time(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  status='starting', initial_gpu=initial, initial_host=initial_host, initial_fans=fans, latest_bmc=bmc,
                  limits=dict(gpu_C=60, hbm_C=65, cpu_C=60, peripheral_C=45, pch_C=55, fan_percent=70),
                  recent_samples=[])
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending = None
    next_bmc = next_fans = next_save = 0.0
    exit_code = 0

    def collect_pending_bmc():
        nonlocal pending, bmc
        if pending is not None and pending.done():
            bmc = pending.result()
            pending = None
            check_bmc(bmc)
            record['latest_bmc'] = bmc

    def check_retry_host():
        check_host(initial_host)
        collect_pending_bmc()

    def probe_event(event):
        record['gpu_monitor'] = dict(probe.stats)
        record['updated_unix'] = time.time()
        write_json(ROOT / 'health.json', record)
        print('GPU monitoring: ' + json.dumps(event), file=sys.stderr, flush=True)

    probe = GpuProbe(watch.gpu_sample, initial, initial_host, record['boot_id'],
                     check_host=check_retry_host,
                     cancelled=lambda: stop_requested or (child is not None and child.poll() is not None),
                     emit=probe_event)
    record['gpu_monitor'] = dict(probe.stats)
    try:
        with (ROOT / 'server.stdout.log').open('a') as stdout, (ROOT / 'server.stderr.log').open('a') as stderr:
            child = subprocess.Popen(config['command'], env=env, stdout=stdout, stderr=stderr,
                                     stdin=subprocess.DEVNULL, user=owner.st_uid, group=owner.st_gid,
                                     start_new_session=True)
            record.update(pid=child.pid, status='running')
            write_json(ROOT / 'health.json', record)
            while child.poll() is None and not stop_requested:
                now = time.monotonic()
                gpu = probe.read()
                cpu, available = check_host(initial_host)
                if now >= next_fans:
                    fans = watch.fan_duty_sample()
                    if fans['mode'] != '00' or max(fans['raw']) > 178:
                        raise RuntimeError('Fan policy exceeded')
                    next_fans = now + 5
                sample = dict(unix=time.time(), gpu=gpu, cpu_C=cpu, fans=fans,
                              available_kib=available, nic=watch.nic_sample())
                record['recent_samples'].append(sample)
                record['recent_samples'] = record['recent_samples'][-30:]
                collect_pending_bmc()
                if now >= next_bmc and pending is None:
                    pending = pool.submit(watch.bmc_sample)
                    next_bmc = now + 30
                if now >= next_save:
                    record['updated_unix'] = time.time()
                    write_json(ROOT / 'health.json', record)
                    next_save = now + 5
                time.sleep(2)
            if not stop_requested:
                exit_code = child.returncode or 0
                record['status'] = 'exited'
            else:
                record['status'] = 'stopped_by_request'
    except ProbeCancelled:
        if stop_requested:
            record['status'] = 'stopped_by_request'
        else:
            record['status'] = 'exited'
            exit_code = child.returncode or 0
    except BaseException as error:
        record.update(status='stopped_on_error', error=repr(error))
        (ROOT / 'STOP_REQUESTED').write_text(repr(error) + '\n')
        print(repr(error), file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        pool.shutdown(wait=True)
        if child is not None:
            record['child_returncode'] = child.returncode
        record['finished_unix'] = time.time()
        record['gpu_monitor'] = dict(probe.stats)
        write_json(ROOT / 'health.json', record)
        lock.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
