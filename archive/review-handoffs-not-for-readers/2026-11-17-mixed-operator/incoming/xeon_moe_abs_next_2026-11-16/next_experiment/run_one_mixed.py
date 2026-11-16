#!/usr/bin/env python3
"""User-machine launcher for one isolated IQ2up mixed-cache process.

Run under the existing outer guard.py for temperature/API-activity/counter guards.
This launcher does not replace those monitors, operate services, or change a CPU
library. It launches exactly the requested process, never the eight-process suite.
The child inherits the outer guard's process group; no new session is created.
Only Python's standard library is required. No real execution was done in review.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

LABELS = ('F', 'E', 'E', 'F', 'E', 'F', 'F', 'E')
CPU_SHA = {
    'F': '33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3',
    'E': '120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3',
}
COMMIT = '31b1292b2767618a28ef6e3f1c9c4450b1fc21be'
INPUT_SHA = '9dc5705e04389d3d08d894ddf25ce675fa2ff3bed73eccc4ba9a156c8028ec93'
PLAN_SHA = 'cf11b82e3e953bbb06a0d5df164dbb6d53f6b19923efaeb79c72ceeef068bee6'
INPUT_BYTES = 163840
CASES = ('C0', 'C6', 'Cplan')
EXPECTED_BYTES = 442368
REQUIRED_FILES = ('raw.jsonl', 'expected-C0.bin', 'expected-C6.bin', 'expected-Cplan.bin')
RESERVE_BYTES = 64 * 1024**3
CACHE_BUDGET_BYTES = 304 * 1024**2
TIMEOUT_SECONDS = 180
INVARIANTS = ('base_library_sha256', 'harness_sha256', 'inputs_sha256', 'plan_sha256',
              'case_spec_sha256', 'case_header_sha256', 'shard_stat', 'extra_library_dirs', 'launcher_sha256')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def nonfinite(value):
    raise ValueError('nonfinite JSON number: ' + value)


def load_json(path, limit=4 * 1024**2):
    require(path.stat().st_size <= limit, 'JSON too large: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=nonfinite)


def file_record(path):
    require(path.is_file(), 'required regular file missing: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def canonical_file(value, executable=False):
    path = Path(value).resolve(strict=True)
    require(stat.S_ISREG(path.stat().st_mode), 'not a regular file: ' + str(path))
    if executable:
        require(os.access(path, os.X_OK), 'file is not executable: ' + str(path))
    return path


def shard_stat(path):
    s = path.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_size > 0, 'shard must be a nonempty regular file')
    return {'path': str(path), 'device': s.st_dev, 'inode': s.st_ino,
            'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def verify_case(case_dir):
    spec_path = canonical_file(case_dir / 'mixed_case.json')
    receipt = load_json(canonical_file(case_dir / 'preparation_receipt.json'))
    spec = load_json(spec_path)
    require(receipt.get('complete') is True, 'case preparation did not complete')
    require(spec.get('schema_version') == 1 and spec.get('commit') == COMMIT, 'wrong case schema/commit')
    require(spec.get('tensor_name') == 'blk.0.ffn_up_exps.weight', 'wrong case tensor')
    require(spec.get('shape') == {'K': 5120, 'M': 2304, 'experts': 384, 'used': 6, 'groups': 8, 'inputs': 1},
            'wrong case geometry')
    require(spec.get('source_plan_file') == 'source-plan.txt', 'case must include source-plan.txt')
    paths = {name: canonical_file(case_dir / name) for name in
             ('mixed_case.json', 'mixed_case_data.h', 'inputs.bin', 'source-plan.txt')}
    records = {name: file_record(path) for name, path in paths.items()}
    for name, observed in records.items():
        require(receipt.get('files', {}).get(name) == observed, 'case receipt mismatch: ' + name)
    require(records['inputs.bin'] == {'bytes': INPUT_BYTES, 'sha256': INPUT_SHA}, 'input fixture bytes/hash differ')
    require(spec.get('inputs_bytes') == INPUT_BYTES and spec.get('inputs_sha256') == INPUT_SHA,
            'case input declaration differs')
    require(spec.get('plan_sha256') == PLAN_SHA and records['source-plan.txt']['sha256'] == PLAN_SHA,
            'source plan bytes/hash differ')
    require(spec.get('output_files', {}).get('source-plan.txt') == records['source-plan.txt'],
            'source plan file declaration differs')
    routes, cache_ids = spec.get('routes'), spec.get('cache_ids')
    require(isinstance(routes, list) and len(routes) == 48 and len(set(routes)) == 48 and
            all(type(x) is int and 0 <= x < 384 for x in routes), 'invalid route IDs')
    require(isinstance(cache_ids, list) and len(cache_ids) == 39 and cache_ids == sorted(set(cache_ids)) and
            set(cache_ids) <= set(routes), 'invalid Cplan cache IDs')
    masks = [sum(1 << j for j in range(6) if routes[g*6+j] in cache_ids) for g in range(8)]
    hits = [m.bit_count() for m in masks]
    require(masks == spec.get('masks') and hits == [5, 5, 5, 5, 3, 5, 6, 5] and
            hits == spec.get('cached_of_six'), 'Cplan mask/count mismatch')
    # Independently bind the selected 39 fixture experts to the exact copied plan row.
    lines = paths['source-plan.txt'].read_text(encoding='utf-8').splitlines()
    target = [line.split() for line in lines if line.split()[:1] == [spec['tensor_name']]]
    require(len(target) == 1 and target[0][1:5] == ['iq2_s', '5120', '2304', '384'], 'plan target row differs')
    selected_count = int(target[0][5]); plan_ids = [int(value) for value in target[0][6:]]
    require(len(plan_ids) == selected_count and len(set(plan_ids)) == selected_count,
            'plan target expert count differs')
    require(sorted(set(plan_ids) & set(routes)) == cache_ids, 'Cplan is not the fixed plan intersection')
    return spec, paths, records


def parse_cpu_list(text):
    result = set()
    for item in text.strip().split(','):
        require(bool(item), 'empty CPU/node list item')
        parts = item.split('-')
        require(len(parts) in (1, 2) and all(x.isdecimal() for x in parts), 'invalid CPU/node list')
        first, last = int(parts[0]), int(parts[-1])
        require(0 <= first <= last <= 65535, 'CPU/node range out of bounds')
        result.update(range(first, last+1))
    return result


def host_preflight():
    require(sys.platform == 'linux' and hasattr(os, 'sched_getaffinity'), 'Linux affinity support required')
    command = shutil.which('numactl')
    require(command is not None, 'numactl is unavailable')
    numactl = canonical_file(command, executable=True)
    allowed = set(os.sched_getaffinity(0))
    target = set(range(24))
    require(target <= allowed, 'CPU affinity must permit the original 0..23 binding')
    online = parse_cpu_list(Path('/sys/devices/system/cpu/online').read_text())
    require(target <= online, 'required logical CPUs 0..23 are not all online')
    topology = {}
    for cpu in sorted(online):
        folder = Path('/sys/devices/system/cpu') / ('cpu'+str(cpu)) / 'topology'
        topology[cpu] = (int((folder/'physical_package_id').read_text()), int((folder/'core_id').read_text()))
    require(len(set(topology.values())) == 12, 'host must have exactly 12 online physical cores')
    require(len({topology[cpu] for cpu in range(12)}) == 12, 'CPUs 0..11 are not twelve distinct physical cores')
    packages = {package for package, _ in topology.values()}
    require(len(packages) == 2 and all(len({pair for pair in topology.values() if pair[0] == p}) == 6 for p in packages),
            'expected two sockets with six online physical cores each')
    nodes = parse_cpu_list(Path('/sys/devices/system/node/online').read_text())
    require({0, 1} <= nodes, 'NUMA nodes 0 and 1 must be online')
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    require({0, 1} <= parse_cpu_list(fields.get('Mems_allowed_list', '')), 'memory policy does not permit nodes 0,1')
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            parts = line.split(); require(len(parts) == 3 and parts[2] == 'kB', 'invalid MemAvailable units')
            available = int(parts[1]) * 1024
    require(available is not None and available >= RESERVE_BYTES + CACHE_BUDGET_BYTES,
            'MemAvailable cannot preserve 64GiB reserve plus 304MiB cache budget')
    return numactl, {'allowed_affinity': sorted(allowed), 'required_affinity': list(range(24)),
                     'pool_cpus': list(range(12)), 'online_cpus': sorted(online),
                     'physical_cores': 12, 'packages': sorted(packages),
                     'target_topology': {str(cpu): list(topology[cpu]) for cpu in range(24)},
                     'numa_nodes_online': sorted(nodes), 'mem_available_bytes': available,
                     'reserve_bytes': RESERVE_BYTES, 'cache_budget_bytes': CACHE_BUDGET_BYTES,
                     'scope': 'Read-only preflight; outer guard required for live API/temperature/counter monitoring.'}


def child_environment(cpu, base, extra):
    prefixes = ('LLAMA_', 'GGML_', 'OMP_', 'GOMP_', 'LD_')
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    dirs = []
    for path in [cpu.parent, base.parent, *extra]:
        require(':' not in str(path), 'library directory contains LD path separator')
        if str(path) not in dirs:
            dirs.append(str(path))
    fixed = {'LD_LIBRARY_PATH': ':'.join(dirs), 'OMP_DYNAMIC': 'FALSE', 'OMP_THREAD_LIMIT': '24',
             'OMP_MAX_ACTIVE_LEVELS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    env.update(fixed)
    receipt = {'controlled_values': fixed, 'removed_prefixes': list(prefixes),
               'other_environment_inherited': True, 'inherits_outer_process_group': True}
    return env, receipt


def inspect_outputs(folder, index, spec, identity):
    records = {name: file_record(folder/name) for name in REQUIRED_FILES}
    for case in CASES:
        require(records['expected-'+case+'.bin']['bytes'] == EXPECTED_BYTES, 'wrong expected output byte count: '+case)
    require(0 < records['raw.jsonl']['bytes'] <= 16*1024**2, 'raw JSONL size outside bounds')
    rows = [json.loads(line, parse_constant=nonfinite) for line in (folder/'raw.jsonl').read_text().splitlines()]
    require(rows and all(isinstance(row, dict) for row in rows), 'invalid raw records')
    meta = [r for r in rows if r.get('kind') == 'mixed_meta']
    complete = [r for r in rows if r.get('kind') == 'mixed_complete']
    cases = [r for r in rows if r.get('kind') == 'mixed_case']
    chunks = [r for r in rows if r.get('kind') == 'mixed_chunk']
    require(len(meta) == 1 and meta[0] == rows[0] and len(complete) == 1 and complete[0] == rows[-1],
            'missing/duplicate/out-of-order metadata or completion')
    require(len(cases) == 3 and len(chunks) == 192 and len(rows) == 197, 'wrong raw inventory')
    require({r.get('case') for r in cases} == set(CASES), 'missing or duplicate cache case')
    m, end = meta[0], complete[0]
    require(m.get('schema') == 1 and m.get('process_index') == index and m.get('library_label') == LABELS[index],
            'raw process identity mismatch')
    require(m.get('commit') == COMMIT and m.get('tensor') == spec['tensor_name'] and m.get('routes') == spec['routes'],
            'compiled case metadata does not match supplied case')
    require(m.get('inputs_sha256') == identity['inputs_sha256'] and m.get('plan_sha256') == identity['plan_sha256'] and
            m.get('hashes_verified_by_harness') is False, 'raw declared hashes or verification scope differ')
    require(m.get('threads') == 12 and m.get('blocks') == 16 and m.get('repeats') == 32 and
            m.get('ggml_numa') is False and m.get('clock') == 'CLOCK_MONOTONIC', 'raw execution settings differ')
    require(m.get('cpu_provider') == identity['cpu_library_path'] and m.get('cache_provider') == identity['cpu_library_path'] and
            m.get('base_provider') == identity['base_library_path'], 'raw provider identity differs')
    require(m.get('mode_provider') == (identity['cpu_library_path'] if LABELS[index] == 'E' else ''),
            'raw abs provider identity differs')
    rotation = (index//2) % 3
    require(m.get('case_order') == list(CASES[rotation:]+CASES[:rotation]), 'wrong case order')
    for row in cases:
        case = row['case']; selection = [] if case == 'C0' else spec['routes'] if case == 'C6' else spec['cache_ids']
        require(row.get('selected_ids') == selection and row.get('cache_bytes') == spec['cache_cases'][case]['cache_payload_bytes'],
                'actual cache selection differs: '+case)
        require(sorted(row.get('pins', [])) == list(range(12)) and row.get('bit_mismatches') == 0,
                'case affinity or correctness failed: '+case)
    require(all(row.get('bit_mismatches') == 0 for row in chunks), 'timed output mismatch')
    require(end.get('passed') is True and end.get('error') is None and end.get('chunks') == 192 and
            end.get('timed_graph_calls') == 6144 and end.get('outputs_bytes') == 3*EXPECTED_BYTES,
            'harness did not complete the required protocol')
    require((folder/'maps.txt').is_file(), 'provider map receipt missing')
    stdout_passed = False
    for line in (folder/'stdout.log').read_text(errors='replace').splitlines():
        try:
            row = json.loads(line, parse_constant=nonfinite)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('passed') is True:
            stdout_passed = True
    require(stdout_passed, 'harness success stdout missing')
    return records


def verify_previous(root, index, identity):
    for previous in range(index):
        folder = root/('process-'+str(previous))
        status = load_json(folder/'STATUS.json')
        require(status.get('schema') == 1 and status.get('passed') is True and status.get('state') == 'completed' and status.get('exit_code') == 0 and
                status.get('error') is None and status.get('process_index') == previous and
                status.get('library_label') == LABELS[previous], 'earlier process is not successful: '+str(previous))
        prior = status.get('identity', {})
        require(prior.get('cpu_library_sha256') == CPU_SHA[LABELS[previous]], 'earlier CPU hash/label mismatch')
        for key in INVARIANTS:
            require(prior.get(key) == identity.get(key), 'run identity changed at '+key)
        for name in REQUIRED_FILES:
            require(file_record(folder/name) == status.get('files', {}).get(name), 'earlier result bytes changed: '+name)


def save_status(folder, record):
    temporary = folder/'STATUS.tmp'
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(folder/'STATUS.json')


def terminate_child(child):
    if child is None or child.poll() is not None:
        return
    try:
        child.terminate()
    except ProcessLookupError:
        child.wait(timeout=5)
        return
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            child.kill()
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def run_one(args):
    index = args.process_index
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'RUN.lock').open('a')
    folder = root/('process-'+str(index))
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        folder.mkdir(exist_ok=False)  # Never reuse or overwrite a failed process directory.
    except BaseException:
        lock.close()
        raise
    record = {'schema': 1, 'process_index': index, 'library_label': LABELS[index],
              'passed': False, 'exit_code': None, 'error': None, 'files': {}, 'identity': {},
              'started_monotonic_ns': time.monotonic_ns(), 'ended_monotonic_ns': None,
              'argv': None, 'env': None, 'timeout_seconds': TIMEOUT_SECONDS,
              'outer_guard_required': True, 'retry_policy': 'Any failed attempt requires a fresh run root.'}
    child = None
    old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, _frame):
        raise InterruptedError('launcher received signal '+str(signum))
    for signum in old_handlers:
        signal.signal(signum, interrupted)
    try:
        allowed = {'RUN.lock', *('process-'+str(i) for i in range(index+1))}
        require(all(p.name in allowed for p in root.iterdir()), 'run root contains later/unrecognized artifacts')
        cpu = canonical_file(args.cpu_library); base = canonical_file(args.base_library)
        harness = canonical_file(args.harness, executable=True); shard = canonical_file(args.shard)
        case_dir = args.case_dir.resolve(strict=True)
        extra = [p.resolve(strict=True) for p in args.library_dir]
        require(all(p.is_dir() for p in extra), 'library search path is not a directory')
        spec, case_paths, case_records = verify_case(case_dir)
        identity = {'cpu_library_sha256': file_record(cpu)['sha256'], 'base_library_sha256': file_record(base)['sha256'],
                    'harness_sha256': file_record(harness)['sha256'], 'inputs_sha256': case_records['inputs.bin']['sha256'],
                    'plan_sha256': case_records['source-plan.txt']['sha256'], 'case_spec_sha256': case_records['mixed_case.json']['sha256'],
                    'case_header_sha256': case_records['mixed_case_data.h']['sha256'], 'cpu_library_path': str(cpu),
                    'base_library_path': str(base), 'harness_path': str(harness), 'case_directory': str(case_dir),
                    'shard_stat': shard_stat(shard), 'extra_library_dirs': [str(p) for p in extra],
                    'launcher_sha256': file_record(Path(__file__).resolve())['sha256']}
        record['identity'] = identity
        require(identity['cpu_library_sha256'] == CPU_SHA[LABELS[index]], 'CPU hash does not match fixed process label')
        verify_previous(root, index, identity)
        numactl, host = host_preflight(); record['host_preflight'] = host
        env, env_receipt = child_environment(cpu, base, extra); record['env'] = env_receipt
        argv = [str(numactl), '--physcpubind=0-23', '--interleave=0,1', str(harness), 'bench', str(shard),
                str(case_paths['inputs.bin']), str(folder), str(index), str(cpu), str(base)]
        record['argv'] = argv
        record['state'] = 'starting'; save_status(folder, record)
        with (folder/'stdout.log').open('xb') as out, (folder/'stderr.log').open('xb') as err:
            # Inherit the outer guard process group. Never use start_new_session=True.
            child = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=False)
            record['child_pid'] = child.pid
            record['exit_code'] = child.wait(timeout=TIMEOUT_SECONDS)
        require(record['exit_code'] == 0, 'harness exited unsuccessfully')
        for name, path in [('cpu_library_sha256', cpu), ('base_library_sha256', base), ('harness_sha256', harness)]:
            require(file_record(path)['sha256'] == identity[name], 'executable/library changed during run: '+name)
        _, _, after_cases = verify_case(case_dir)
        require(after_cases == case_records and shard_stat(shard) == identity['shard_stat'], 'case/shard changed during run')
        record['files'] = inspect_outputs(folder, index, spec, identity)
        record['auxiliary_files'] = {name: file_record(folder/name) for name in ('maps.txt', 'stdout.log', 'stderr.log')}
        record['passed'] = True
    except BaseException as error:
        record['error'] = type(error).__name__+': '+str(error)
    finally:
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            try:
                terminate_child(child)
            except BaseException as error:
                record['error'] = (record['error']+'; ' if record['error'] else '')+'child cleanup failed: '+str(error)
                record['passed'] = False
            if child is not None:
                record['exit_code'] = child.returncode
            # Preserve hashes of partial output; never delete it or turn it into success.
            for name in REQUIRED_FILES:
                if (folder/name).is_file():
                    try:
                        record['files'][name] = file_record(folder/name)
                    except OSError as error:
                        record['error'] = record['error'] or ('partial file hashing failed: '+str(error))
                        record['passed'] = False
            record['ended_monotonic_ns'] = time.monotonic_ns()
            record['state'] = 'completed' if record['passed'] else 'failed'
            save_status(folder, record)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            lock.close()
    print(json.dumps({'process_directory': str(folder), 'passed': record['passed'],
                      'exit_code': record['exit_code'], 'error': record['error']}, ensure_ascii=False))
    return 0 if record['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('harness', 'cpu-library', 'base-library', 'shard', 'case-dir', 'run-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, action='append', default=[])
    parser.add_argument('--process-index', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    try:
        return run_one(args)
    except (OSError, RuntimeError, ValueError) as error:
        print('FAIL without overwriting an existing process: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
