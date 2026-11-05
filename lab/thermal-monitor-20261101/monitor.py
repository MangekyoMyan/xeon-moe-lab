#!/usr/bin/env python3
"""Local-only telemetry and incident outbox. Never starts/stops models or changes fans."""
import argparse
import concurrent.futures
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.chmod(tmp, 0o644)
    tmp.replace(path)

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default

def rotating_log(path, value, limit=5 * 1024**2):
    if path.exists() and path.stat().st_size >= limit:
        for i in range(2, 0, -1):
            src = Path(str(path) + f'.{i}')
            if src.exists():
                src.replace(Path(str(path) + f'.{i+1}'))
        path.replace(Path(str(path) + '.1'))
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')

def command(args, env=None, timeout=8):
    p = subprocess.run(args, env=env, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{Path(args[0]).name}: exit {p.returncode}')
    return p.stdout

def number(value):
    if value is None or value.strip() in ('N/A', 'na', ''):
        return None
    v = float(value.split()[0])
    return v if math.isfinite(v) else None

def sample_cpu():
    values = {}
    for hw in Path('/sys/class/hwmon').glob('hwmon*'):
        if (hw / 'name').read_text().strip() != 'coretemp':
            continue
        for path in hw.glob('temp*_input'):
            label = path.with_name(path.name.replace('_input', '_label'))
            name = label.read_text().strip() if label.exists() else path.name
            values[hw.name + ':' + name] = float(path.read_text()) / 1000
    if not values:
        raise RuntimeError('No coretemp sensors')
    return values

def sample_gpu(config):
    root = ET.fromstring(command(['nvidia-smi', '-i', config['gpu_uuid'], '-q', '-x']))
    gpu = root.find('gpu')
    if gpu is None or gpu.findtext('uuid') != config['gpu_uuid']:
        raise RuntimeError('GPU identity mismatch')
    value = lambda key: number(gpu.findtext(key))
    return {'gpu_C': value('temperature/gpu_temp'), 'hbm_C': value('temperature/memory_temp'),
            'power_W': value('gpu_power_readings/instant_power_draw'),
            'power_limit_W': value('gpu_power_readings/current_power_limit'),
            'used_MiB': value('fb_memory_usage/used'), 'util_percent': value('utilization/gpu_util'),
            'ecc_mode': gpu.findtext('ecc_mode/current_ecc'),
            'ecc_single': value('ecc_errors/volatile/single_bit/total'),
            'ecc_double': value('ecc_errors/volatile/double_bit/total'),
            'replays': value('pci/replay_counter')}

def parse_bmc(text):
    rows = {}
    for line in text.splitlines():
        fields = [s.strip() for s in line.split('|')]
        if len(fields) < 4:
            continue
        try:
            value = number(fields[1])
        except ValueError:
            value = None
        rows[fields[0]] = {'value': value, 'unit': fields[2], 'status': fields[3]}
    return rows

def sample_bmc(config):
    env = dict(os.environ, LD_LIBRARY_PATH=config['ipmi_library_path'], LC_ALL='C')
    def ipmi(*args):
        return command([config['ipmitool'], '-I', 'open', *args], env=env, timeout=12)
    rows = parse_bmc(ipmi('sensor', 'list'))
    dimms = {k: v['value'] for k, v in rows.items() if k.startswith(('P1-DIMM', 'P2-DIMM')) and k.endswith(' TEMP')}
    fans = {k: rows.get(k, {}).get('value') for k in config['expected_fans']}
    duty = [int(ipmi('raw', '0x30', '0x90', '0x5a', '0x03', hex(r), '0x01').strip(), 16) for r in range(0x10, 0x14)]
    if any(v < 0 or v > 255 for v in duty):
        raise RuntimeError('Invalid fan duty')
    return {'observed_unix': time.time(), 'dimms_C': dimms, 'fan_rpm': fans,
            'fan_percent': [v * 100 / 255 for v in duty], 'fan_mode': ipmi('raw', '0x30', '0x45', '0x00').strip(),
            'pch_C': rows.get('PCH Temp', {}).get('value'),
            'peripheral_C': rows.get('Peripheral Temp', {}).get('value'),
            'system_C': rows.get('System Temp', {}).get('value'),
            'bad_sensor_status': {k: v['status'] for k, v in rows.items() if v['status'] not in ('ok', 'na')}}

def sample_host_counters():
    paths = list(Path('/sys/devices/system/edac/mc').glob('mc[0-9]*/*count'))
    paths += list(Path('/sys/devices/system/cpu').glob('cpu*/thermal_throttle/*throttle_count'))
    values = {str(p): int(p.read_text()) for p in paths}
    if not any('/edac/' in k for k in values):
        raise RuntimeError('EDAC counters unavailable')
    return values

class Detector:
    """Persistent latches, hysteresis and counter deltas; no commands or notifications."""
    def __init__(self, config, state=None):
        self.c = config
        self.s = state or {'alarms': {}, 'last_counter': {}, 'failures': {}, 'boot_id': None}

    def update(self, sample):
        now = sample['unix']
        if self.s.get('boot_id') != sample['boot_id']:
            self.s['boot_id'] = sample['boot_id']
            self.s['last_counter'] = {}
            self.s['counter_notified'] = {}
        observations = {}
        def condition(key, bad, recovered, description, value=None, limit=None):
            observations[key] = dict(bad=bad, recovered=recovered, description=description, value=value, limit=limit)
        def temp(key, value, limit, description):
            if value is None:
                return
            condition(key, value >= limit, value < limit - self.c['recovery_margin'], description, value, limit)
        cpu = sample.get('cpu', {})
        if cpu:
            temp('cpu_C', max(cpu.values()), self.c['limits']['cpu_C'], 'CPU温度')
        gpu = sample.get('gpu', {})
        for key, label in [('gpu_C', 'GPU温度'), ('hbm_C', 'GPUメモリ温度')]:
            temp(key, gpu.get(key), self.c['limits'][key], label)
        if gpu:
            condition('gpu_ecc_disabled', gpu.get('ecc_mode') != 'Enabled', gpu.get('ecc_mode') == 'Enabled', 'GPU ECC設定', gpu.get('ecc_mode'))
        api = sample.get('api_guard', {})
        if api.get('boot_id') == sample['boot_id']:
            failed = api.get('status') == 'stopped_on_error'
            condition('api_guard_stopped', failed, not failed, '既存API保護機能による停止', api.get('error'))
        for path, api in sample.get('additional_api_guards', {}).items():
            if api.get('boot_id') == sample['boot_id']:
                failed = api.get('status') == 'stopped_on_error'
                condition('api_guard:' + path, failed, not failed, 'API保護機能による停止: ' + path, api.get('error'))
        bmc = sample.get('bmc')
        if bmc:
            for key, value in bmc['dimms_C'].items():
                temp('dimm:' + key, value, self.c['limits']['dimm_C'], key)
            for key, label in [('pch_C', 'PCH温度'), ('peripheral_C', '周辺部温度')]:
                temp(key, bmc.get(key), self.c['limits'][key], label)
            for fan, value in bmc['fan_rpm'].items():
                if value is not None:
                    condition('fan:' + fan, value <= 0, value > 0, fan + '回転数', value, 0)
            if bmc['fan_percent']:
                value = max(bmc['fan_percent'])
                condition('fan_percent', value > self.c['limits']['fan_percent'], value < self.c['limits']['fan_percent'] - 3,
                          'ファン出力', value, self.c['limits']['fan_percent'])
            condition('fan_mode', bmc['fan_mode'] != '00', bmc['fan_mode'] == '00', 'Standard以外のファンモード', bmc['fan_mode'])
        failures = dict(sample.get('errors', {}))
        if gpu and (gpu.get('gpu_C') is None or gpu.get('hbm_C') is None):
            failures['gpu_temperature'] = 'GPU/HBM temperature missing'
        if bmc:
            if sum(v is not None for v in bmc['dimms_C'].values()) != self.c['expected_dimm_count']:
                failures['dimm_temperature'] = 'Not all16 DIMM temperatures available'
            if any(v is None for v in bmc['fan_rpm'].values()):
                failures['fan_rpm'] = 'Expected fan sensor unavailable'
            if bmc.get('pch_C') is None or bmc.get('peripheral_C') is None:
                failures['board_temperature'] = 'Board temperatures unavailable'
        for key in set(failures) | set(self.s['failures']):
            count = self.s['failures'].get(key, 0) + 1 if key in failures else 0
            self.s['failures'][key] = count
            condition('sensor:' + key, count >= self.c['sensor_failure_samples'], count == 0,
                      'センサー取得失敗: ' + key, failures.get(key))
        all_counters = dict(sample.get('host_counters', {}))
        all_counters.update({'gpu:' + k: gpu[k] for k in ['ecc_single', 'ecc_double', 'replays'] if gpu.get(k) is not None})
        counter_events = []
        notified = self.s.setdefault('counter_notified', {})
        for key, value in all_counters.items():
            previous = self.s['last_counter'].get(key, 0)
            if value > previous and now - notified.get(key, -1e20) >= self.c['repeat_cooldown_seconds']:
                counter_events.append({'key': key, 'description': 'ECC/熱制限/PCIeカウンター増加', 'value': value,
                                       'previous': previous, 'delta': value - previous})
                notified[key] = now
            self.s['last_counter'][key] = value
        opened = []
        recovered = []
        for key, obs in observations.items():
            alarm = self.s['alarms'].setdefault(key, {'active': False, 'recovery_count': 0, 'last_notified': -1e20})
            if obs['bad']:
                alarm['recovery_count'] = 0
                if not alarm['active']:
                    alarm.update(active=True, since=now, notified=False)
                alarm['observation'] = obs
                if not alarm.get('notified') and now - alarm['last_notified'] >= self.c['repeat_cooldown_seconds']:
                    opened.append(dict(key=key, **obs))
                    alarm.update(notified=True, last_notified=now)
            elif alarm['active'] and obs['recovered']:
                alarm['recovery_count'] += 1
                if alarm['recovery_count'] >= self.c['recovery_samples']:
                    alarm.update(active=False, recovery_count=0, recovered_unix=now)
                    recovered.append(key)
            else:
                alarm['recovery_count'] = 0
        return opened + counter_events, recovered

def emit_event(root, items, sample):
    event = {'id': uuid.uuid4().hex, 'unix': sample['unix'], 'boot_id': sample['boot_id'],
             'kind': 'hardware_alert', 'items': items, 'snapshot_path': str(root / 'latest.json')}
    atomic(root / 'outbox' / (event['id'] + '.json'), event)
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return event

def collect(config, bmc=None, bmc_error=None):
    s = {'unix': time.time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'errors': {}}
    for name, fn in [('cpu', sample_cpu), ('gpu', lambda: sample_gpu(config)), ('host_counters', sample_host_counters)]:
        try:
            s[name] = fn()
        except (OSError, ValueError, RuntimeError, ET.ParseError, subprocess.SubprocessError) as e:
            s['errors'][name] = type(e).__name__ + ': ' + str(e)[:160]
    if bmc is not None:
        s['bmc'] = bmc
    if bmc_error:
        s['errors']['bmc'] = bmc_error
    elif bmc is None or s['unix'] - bmc['observed_unix'] > 3 * config['bmc_seconds']:
        s['errors']['bmc'] = 'BMC sample unavailable or stale'
    try:
        api = read_json(Path(config['workspace']) / 'llm-lab/api-server-20261030/health.json', {})
        s['api_guard'] = {k: api.get(k) for k in ['boot_id', 'status', 'error', 'finished_unix']}
    except (OSError, ValueError):
        # The API may be absent or inactive; it is not required for hardware monitoring.
        pass
    s['additional_api_guards'] = {}
    for path in config.get('additional_api_health_paths', []):
        try:
            api = read_json(Path(path), {})
            s['additional_api_guards'][path] = {k: api.get(k) for k in ['boot_id', 'status', 'error', 'finished_unix']}
        except (OSError, ValueError):
            pass
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=HERE / 'config.json')
    ap.add_argument('--state-dir', type=Path)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    c = read_json(args.config, None)
    root = args.state_dir or Path(c['state_dir'])
    root.mkdir(parents=True, exist_ok=True)
    (root / 'outbox').mkdir(exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        detector = Detector(c, read_json(root / 'detector.json', None))
        stop = False
        def stopped(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGTERM, stopped)
        signal.signal(signal.SIGINT, stopped)
        pending = None
        bmc = None
        bmc_error = None
        next_bmc = next_log = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            while not stop:
                start = time.monotonic()
                if args.once:
                    try:
                        bmc = sample_bmc(c)
                    except Exception as e:
                        bmc_error = str(e)[:160]
                else:
                    if pending is not None and pending.done():
                        try:
                            bmc = pending.result()
                            bmc_error = None
                        except Exception as e:
                            bmc_error = str(e)[:160]
                        pending = None
                    if start >= next_bmc and pending is None:
                        pending = pool.submit(sample_bmc, c)
                        next_bmc = start + c['bmc_seconds']
                s = collect(c, bmc, bmc_error)
                alerts, cleared = detector.update(s)
                s['active_alerts'] = {k: v for k, v in detector.s['alarms'].items() if v['active']}
                s['monitor'] = {'pid': os.getpid(), 'limits': c['limits'], 'sample_seconds': c['sample_seconds'],
                                'bmc_seconds': c['bmc_seconds'], 'automatic_model_actions': False}
                atomic(root / 'latest.json', s)
                # Persist outbox before latches: a crash may duplicate an incident, but cannot silently lose it.
                if alerts:
                    emit_event(root, alerts, s)
                atomic(root / 'detector.json', detector.s)
                if start >= next_log or alerts or cleared:
                    rotating_log(root / 'history.jsonl', s)
                    next_log = start + c['log_seconds']
                if args.once:
                    print(json.dumps({'snapshot': str(root / 'latest.json'), 'alerts': len(alerts), 'errors': s['errors']}))
                    break
                while not stop and time.monotonic() - start < c['sample_seconds']:
                    time.sleep(min(0.5, c['sample_seconds']))

if __name__ == '__main__':
    main()
