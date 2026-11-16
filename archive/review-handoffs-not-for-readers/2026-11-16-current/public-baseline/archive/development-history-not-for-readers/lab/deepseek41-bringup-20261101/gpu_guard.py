"""Bounded retry for GPU-query timeouts, not for actual hardware faults."""
import json
import math
from pathlib import Path
import subprocess
import time

WATCH = Path('/var/lib/xeon-watch/latest.json')
ATTEMPTS = 3
QUERY_SECONDS = 5
RECOVERY_SECONDS = 20
SNAPSHOT_SECONDS = 20
EXPECTED_FANS = {'FAN1', 'FAN3', 'FAN5', 'FANB'}


class ProbeCancelled(Exception):
    pass


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError('Invalid sensor: ' + name)
    return value


def temperatures(values, limit, name):
    if not isinstance(values, dict) or not values:
        raise RuntimeError('Missing sensors: ' + name)
    for key, value in values.items():
        if number(value, name + '/' + key) >= limit:
            raise RuntimeError(name + ' thermal cutoff')


def validate_gpu(gpu, initial):
    if number(gpu.get('gpu_C'), 'GPU') >= 60 or number(gpu.get('hbm_C'), 'HBM') >= 65:
        raise RuntimeError('GPU/HBM thermal cutoff')
    if abs(number(gpu.get('limit_W'), 'GPU power limit') - 150) >= .5 or gpu.get('ecc') != 'Enabled':
        raise RuntimeError('GPU operating policy changed')
    for key in ['ecc_single', 'ecc_double', 'replays']:
        if number(gpu.get(key), key) != number(initial.get(key), 'initial ' + key):
            raise RuntimeError('GPU error counter changed: ' + key)


def read_snapshot():
    return json.loads(WATCH.read_text())


def validate_snapshot(snapshot, initial_gpu, initial_host, boot_id, now, allowed_alert=None):
    age = now - number(snapshot.get('unix'), 'monitor time')
    if not 0 <= age <= SNAPSHOT_SECONDS or snapshot.get('boot_id') != boot_id:
        raise RuntimeError('Independent monitor stale, future-dated or from another boot')
    if snapshot.get('errors') != {} or not isinstance(snapshot.get('active_alerts'), dict):
        raise RuntimeError('Independent monitor unavailable or reporting sensor errors')
    alerts = set(snapshot['active_alerts'])
    if alerts - ({allowed_alert} if allowed_alert else set()):
        raise RuntimeError('Independent monitor reports active alarms')
    g = snapshot.get('gpu', {})
    mapped = dict(g, limit_W=g.get('power_limit_W'), ecc=g.get('ecc_mode'))
    validate_gpu(mapped, initial_gpu)
    temperatures(snapshot.get('cpu'), 60, 'CPU')
    if not initial_host or snapshot.get('host_counters') != initial_host:
        raise RuntimeError('Independent host ECC/throttle counters changed or missing')
    bmc = snapshot.get('bmc', {})
    if not 0 <= now - number(bmc.get('observed_unix'), 'BMC time') <= 90:
        raise RuntimeError('Independent BMC snapshot stale or future-dated')
    if number(bmc.get('peripheral_C'), 'Peripheral') >= 45 or number(bmc.get('pch_C'), 'PCH') >= 55:
        raise RuntimeError('Independent board thermal cutoff')
    dimms = bmc.get('dimms_C', {})
    if len(dimms) != 16:
        raise RuntimeError('Independent DIMM sensors incomplete')
    temperatures(dimms, 70, 'DIMM')
    duties = bmc.get('fan_percent', [])
    if bmc.get('fan_mode') != '00' or len(duties) != 4 or any(not 0 <= number(x, 'Fan duty') <= 70 for x in duties):
        raise RuntimeError('Independent fan policy mismatch')
    fans = bmc.get('fan_rpm', {})
    if set(fans) != EXPECTED_FANS or any(number(x, 'Fan RPM') <= 0 for x in fans.values()) or bmc.get('bad_sensor_status') != {}:
        raise RuntimeError('Independent BMC sensor/fan fault')
    return age


class GpuProbe:
    def __init__(self, sample, initial_gpu, initial_host, boot_id, check_host,
                 cancelled=lambda: False, emit=lambda event: None,
                 snapshot=read_snapshot, monotonic=time.monotonic, wall=time.time, sleep=time.sleep):
        self.sample = sample
        self.initial_gpu = initial_gpu
        self.initial_host = initial_host
        self.boot_id = boot_id
        self.check_host = check_host
        self.cancelled = cancelled
        self.emit = emit
        self.snapshot = snapshot
        self.monotonic = monotonic
        self.wall = wall
        self.sleep = sleep
        self.stats = {'timeout_count': 0, 'recovered_queries': 0, 'last_event': None,
                      'max_attempts': ATTEMPTS, 'query_timeout_seconds': QUERY_SECONDS,
                      'recovery_budget_seconds': RECOVERY_SECONDS}

    def check_cancelled(self):
        if self.cancelled():
            raise ProbeCancelled()

    def event(self, kind, attempt, start, **extra):
        event = dict(kind=kind, unix=self.wall(), attempt=attempt,
                     elapsed_seconds=self.monotonic()-start, **extra)
        self.stats['last_event'] = event
        self.emit(event)

    def recovery_checks(self):
        self.check_cancelled()
        self.check_host()
        self.check_cancelled()
        age = validate_snapshot(self.snapshot(), self.initial_gpu, self.initial_host, self.boot_id, self.wall())
        self.check_cancelled()
        return age

    def read(self):
        start = self.monotonic()
        for attempt in range(1, ATTEMPTS+1):
            self.check_cancelled()
            if attempt > 1:
                self.recovery_checks()
                if self.monotonic()-start > RECOVERY_SECONDS-QUERY_SECONDS:
                    raise RuntimeError('GPU monitoring recovery budget exhausted')
            try:
                gpu = self.sample()
            except subprocess.TimeoutExpired as error:
                self.check_cancelled()
                self.stats['timeout_count'] += 1
                self.event('timeout', attempt, start)
                if attempt == ATTEMPTS or self.monotonic()-start >= RECOVERY_SECONDS:
                    raise RuntimeError('GPU monitoring unavailable after bounded retries') from error
                age = self.recovery_checks()
                self.event('retry', attempt, start, independent_snapshot_age_seconds=age)
                # Short sleeps keep an explicit service-stop request responsive.
                until = self.monotonic()+1
                while self.monotonic() < until:
                    self.check_cancelled()
                    self.sleep(min(.1, until-self.monotonic()))
                continue
            validate_gpu(gpu, self.initial_gpu)
            self.check_cancelled()
            if self.monotonic()-start > RECOVERY_SECONDS:
                raise RuntimeError('GPU query exceeded monitoring recovery budget')
            if attempt > 1:
                # The direct query must recover too; the snapshot never substitutes for it.
                self.recovery_checks()
                if self.monotonic()-start > RECOVERY_SECONDS:
                    raise RuntimeError('GPU monitoring recovery budget exhausted')
                self.stats['recovered_queries'] += 1
                self.event('recovered', attempt, start)
            return gpu
        raise AssertionError('Unreachable GPU retry state')
