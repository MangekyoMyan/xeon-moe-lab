#!/usr/bin/env python3
"""Observe real CPU expert graphs between acknowledged measurement boundaries."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results/mxfp4-opt-20261023'
sys.path.insert(0, str(LAB / 'server-study-20261023'))
from run_memory_study import Counter, environment, power, read, sample


def prepare():
    code = (ROOT / 'graph-probe.cpp').read_text()
    code = code.replace('pages[i]=memory+((size_t(i)*7919)%total)*4096;', 'pages[i]=(void*)((uintptr_t(memory)&~uintptr_t(4095))+((size_t(i)*7919)%total)*4096);')
    code = code.replace('#include "ggml-cpu.h"', '#include "ggml-cpu.h"\n#include "ggml-alloc.h"\n#include "ggml-backend.h"\n#include "ggml-cpu/repack.h"')
    anchor = '    placement(memory,bytes);'
    replacement = '''    ggml_context *packed_ctx=nullptr;
    ggml_backend_buffer_t packed_buffer=nullptr;
    ggml_backend_t packed_backend=nullptr;
    const char *packed_env=std::getenv("GGML_CPU_MXFP4_PACKED");
    if(packed_env && !std::strcmp(packed_env,"1")){
        packed_backend=ggml_backend_cpu_init();assert(packed_backend);
        packed_ctx=ggml_init({1024*1024,nullptr,true});assert(packed_ctx);
        ggml_tensor *packed_weights[3];
        for(int phase=0;phase<3;++phase){
            packed_weights[phase]=ggml_new_tensor_3d(packed_ctx,GGML_TYPE_MXFP4,weights[phase]->ne[0],weights[phase]->ne[1],weights[phase]->ne[2]);
        }
        packed_buffer=ggml_backend_alloc_ctx_tensors_from_buft(packed_ctx,ggml_backend_cpu_repack_buffer_type());assert(packed_buffer);
        for(int phase=0;phase<3;++phase){
            ggml_backend_tensor_set(packed_weights[phase],weights[phase]->data,0,ggml_nbytes(weights[phase]));
            weights[phase]=packed_weights[phase];
        }
        assert(!munmap(memory,bytes));memory=nullptr;
        printf("{\\\"packed_weights\\\":true}\\n");fflush(stdout);
    }
    placement((char*)weights[0]->data,bytes);'''
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    anchor = 'ggml_threadpool_free(pool);ggml_free(ctx);ggml_free(wctx);munmap(memory,bytes);'
    replacement = 'ggml_threadpool_free(pool);ggml_free(ctx);ggml_free(wctx);if(memory)munmap(memory,bytes);if(packed_buffer)ggml_backend_buffer_free(packed_buffer);if(packed_ctx)ggml_free(packed_ctx);if(packed_backend)ggml_backend_free(packed_backend);'
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    anchor = '    const double start=now();uint64_t calls=0;'
    assert code.count(anchor) == 1
    code = code.replace(anchor, "    assert(std::getchar()=='G');\n" + anchor)
    anchor = '    // Keep the PID and workers alive until observers have disabled counters.'
    code = code.replace(anchor, "    assert(std::getchar()=='Q');\n" + anchor)
    source = ROOT / 'graph-observe.cpp'
    source.write_text(code)
    commands = []
    for name, base in [('baseline', LAB / 'review-glm-q8-columns4-20261020'), ('candidate', LAB / 'review-mxfp4-20261023'), ('packed', LAB / 'review-mxfp4-packed-20261023')]:
        command = ['c++', '-O3', '-march=ivybridge', '-std=c++17', '-fopenmp', '-I' + str(base / 'source/ggml/include'), '-I' + str(base / 'source/ggml/src'), str(source), '-L' + str(base / 'build/bin'), '-Wl,-rpath,' + str(base / 'build/bin'), '-lggml-cpu', '-lggml-base', '-o', str(ROOT / ('graph-observe-' + name))]
        subprocess.run(command, check=True)
        commands.append(command)
    (OUT / 'graph-observe-build.json').write_text(json.dumps(commands, indent=2) + '\n')


def observe(variant, threads, selected):
    label = f'graph-counters-{variant}-t{threads}-s{selected}'
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    binary = ROOT / ('graph-observe-candidate' if variant == 'paired' else 'graph-observe-' + variant)
    env = dict(environment(), GGML_CPU_MXFP4_ROWS='4' if variant == 'paired' else '0', GGML_CPU_MXFP4_PAIR='1' if variant == 'paired' else '0', GGML_CPU_MXFP4_PACKED='1' if variant == 'packed' else '0')
    command = ['python3', str(LAB / 'without_thp.py'), 'numactl', '--physcpubind=0-23', '--interleave=0,1', str(binary), str(threads), str(selected), '64', 'each', '12', 'both', str(folder / 'tensors.bin')]
    report = {'label': label, 'command': command, 'environment': {k: v for k, v in env.items() if k.startswith(('GGML_', 'OMP_'))}, 'valid': False, 'telemetry': [], 'notes': ['Real CPU graph, not GPU simulation.', 'Counters cover a small handshake interval around the timed loop.', 'IMC counts include other host activity; RAPL is package+DRAM, not whole-system power.']}
    counters = []
    child = None
    turbo = None
    turbo_log = None
    stop = threading.Event()
    monitor = None
    errors = []

    def monitor_host():
        while not stop.is_set():
            try:
                row = sample()
                report['telemetry'].append(row)
                if row['temperature_C'] and max(row['temperature_C'].values()) >= 80:
                    errors.append('Temperature cutoff')
                    os.killpg(child.pid, signal.SIGTERM)
                    return
            except Exception as error:
                errors.append(repr(error))
            stop.wait(.5)

    print('Starting ' + label, flush=True)
    try:
        with (folder / 'stdout.txt').open('w') as stdout, (folder / 'stderr.txt').open('w') as stderr:
            child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1, start_new_session=True)
            report['pid'] = child.pid
            for line in child.stdout:
                stdout.write(line)
                stdout.flush()
                row = json.loads(line)
                if row.get('ready'):
                    report['ready'] = row
                    break
            assert 'ready' in report
            report['tensor_sha256'] = hashlib.sha256((folder / 'tensors.bin').read_bytes()).hexdigest()
            events = ['task-clock', 'cycles:u', 'instructions:u', 'context-switches', 'cpu-migrations', 'minor-faults', 'major-faults', 'cycle_activity.stalls_total:u', 'cycle_activity.stalls_l2_pending:u']
            counters.append(Counter('core', ['-p', str(child.pid), '-e', ','.join(events)], folder))
            counters.append(Counter('imc', ['-a', '-A', '-e', 'uncore_imc_*/event=0x04,umask=0x03/,uncore_imc_*/event=0x04,umask=0x0c/'], folder))
            turbo_log = (folder / 'turbostat.stderr').open('w')
            turbo_command = ['turbostat', '--quiet', '--Summary', '--interval', '1', '--show', 'Busy%,Bzy_MHz,PkgWatt,RAMWatt,PkgTmp,CoreTmp', '--out', str(folder / 'turbostat.tsv')]
            turbo = subprocess.Popen(turbo_command, stdout=turbo_log, stderr=turbo_log, start_new_session=True)
            report['turbostat_command'] = turbo_command
            for counter in counters:
                counter.control('enable')
            report['before'] = sample()
            report['io_before'] = read(f'/proc/{child.pid}/io')
            monitor = threading.Thread(target=monitor_host)
            monitor.start()
            child.stdin.write('G')
            child.stdin.flush()
            line = child.stdout.readline()
            stdout.write(line)
            stdout.flush()
            result = json.loads(line)
            assert result.get('done')
            report['result'] = result
            report['after'] = sample()
            report['io_after'] = read(f'/proc/{child.pid}/io')
            for counter in counters:
                counter.control('disable')
            stop.set()
            monitor.join(timeout=5)
            child.stdin.write('Q')
            child.stdin.flush()
            tail = child.stdout.read()
            stdout.write(tail)
            assert '"post_affinity_valid":true' in tail
            assert child.wait(timeout=10) == 0 and not errors
            if turbo.poll() is None:
                turbo.send_signal(signal.SIGINT)
            turbo.wait(timeout=5)
            report['counters'] = []
            for counter in counters:
                counter.close()
                report['counters'].append({'name': counter.name, 'command': counter.cmd, 'control': counter.events, 'exit_code': counter.proc.returncode})
            counters = []
            report['power'] = power(report['before'], report['after'])
            for name in ('core', 'imc'):
                csv = (folder / (name + '.csv')).read_text()
                assert '<not counted>' not in csv and '<not supported>' not in csv
            report['valid'] = True
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=5)
        for counter in counters:
            counter.close()
        if turbo and turbo.poll() is None:
            turbo.send_signal(signal.SIGINT)
            try:
                turbo.wait(timeout=5)
            except subprocess.TimeoutExpired:
                turbo.kill()
                turbo.wait()
        if turbo_log:
            turbo_log.close()
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        report['observer_errors'] = errors
        (folder / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'done': label, 'valid': report['valid']}), flush=True)
    return report['tensor_sha256']


def main():
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        for threads in (12, 24):
            for selected in (1, 6):
                hashes = []
                for variant in ('baseline', 'paired', 'packed'):
                    hashes.append(observe(variant, threads, selected))
                    assert len(set(hashes)) == 1


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for folder in (ROOT, OUT):
                for path in [folder, *folder.rglob('*')]:
                    os.chown(path, owner.st_uid, owner.st_gid)
