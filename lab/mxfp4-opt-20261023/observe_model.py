#!/usr/bin/env python3
"""Collect diagnostic CPU/DRAM and sampled-symbol profiles in real decode loops."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import run_followup as study

ROOT, LAB, OUT = study.ROOT, study.LAB, study.OUT
FINAL = LAB / 'review-mxfp4-final-20261023/build/bin/llama-bench'


def main():
    phase = ROOT / 'observe_model_phase.py'
    code = (LAB / 'observe_glm_phase.py').read_text()
    assert code.count('"-lm", "mmap"') == 1
    code = code.replace('"-lm", "mmap"', '"-lm", "none"')
    code = code.replace('"--repack", "0"', '"--repack", os.environ.get("LLM_STUDY_REPACK", "0")')
    phase.write_text(code)
    wrapper = ROOT / 'observe_model_counters.py'
    wrapper.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,' + repr(str(LAB)) + ')\nimport observe_glm_counters_light as observer\nobserver.HELPER=Path(' + repr(str(phase)) + ')\nobserver.main()\n')
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = study.audit()
        for variant, mode in [('baseline', 'stat'), ('optimized', 'stat'), ('optimized', 'record'), ('baseline', 'record')]:
            binary = study.BASE / 'llama-bench' if variant == 'baseline' else FINAL
            env = study.environment()
            env['LLM_STUDY_REPACK'] = '0' if variant == 'baseline' else '1'
            if variant != 'baseline':
                env.update(GGML_CPU_MXFP4_PACKED='1', GGML_CPU_MXFP4_COLS4='1', GGML_CPU_MXFP4_PARALLEL_PACK='1', GGML_CPU_REPACK_REUSE_BUFFER='1')
            target = OUT / f'model-profile-{variant}-{mode}'
            if mode == 'stat':
                command = ['python3', str(wrapper), 'decode', 'stalls', '--bench', str(binary), '--model', str(study.MODEL), '--threads', '24', '--tokens', '512', '--window', '18', '--output', str(target)]
            else:
                command = ['python3', str(phase), '--bench', str(binary), '--model', str(study.MODEL), '--threads', '24', '--tokens', '512', '--phase', 'decode', '--mode', 'record', '--seconds', '18', '--cpus', '0-23', '--nodes', '0,1', '--output', str(target)]
            command = ['python3', str(LAB / 'without_thp.py')] + command
            record = {'command': command, 'environment': {k: v for k, v in env.items() if k.startswith(('GGML_', 'OMP_', 'GOMP_', 'LLM_STUDY_'))}}
            (OUT / (target.name + '-command.json')).write_text(json.dumps(record, indent=2) + '\n')
            print('Profiling ' + target.name, flush=True)
            with (OUT / (target.name + '-driver.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=900)
            capture = json.loads((target / 'capture.json').read_text())
            assert capture['valid_sample']
            if mode == 'record':
                with (target / 'symbols.txt').open('w') as log:
                    subprocess.run(['perf', 'report', '--stdio', '--no-children', '--percent-limit', '0.5', '--sort', 'dso,symbol', '-i', str(target / 'profile.data')], stdout=log, stderr=subprocess.STDOUT, check=True)
        assert before == study.audit()
        (OUT / 'model-profiles-complete.json').write_text(json.dumps({'complete': True, 'diagnostic_only': True}) + '\n')


if __name__ == '__main__':
    try:
        main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [OUT, *OUT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid)
