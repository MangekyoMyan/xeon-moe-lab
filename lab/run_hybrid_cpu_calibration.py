#!/usr/bin/env python3
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import run_quantize_tradeoff as runner

LAB = Path(__file__).resolve().parent
OUT = LAB / 'results/hybrid-sim-20261023'
SRC = LAB / 'hybrid-sim-20261023'
LLAMA = LAB / 'review-glm-q8-columns4-20261020'


def main():
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=True)
        source = SRC / 'cpu-expert-calibrate.cpp'
        binary = SRC / 'cpu-expert-calibrate'
        lib = LLAMA / 'build/bin'
        before = hashlib.sha256((lib / 'libggml-cpu.so').read_bytes()).hexdigest()
        cmd = ['c++','-O3','-march=ivybridge','-std=c++17','-fopenmp',
               '-I'+str(LLAMA/'source/ggml/include'),str(source),'-L'+str(lib),
               '-Wl,-rpath,'+str(lib),'-lggml-cpu','-lggml-base','-o',str(binary)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        (OUT/'calibration-build.json').write_text(json.dumps({'command':cmd,'exit_code':build.returncode,
            'stdout':build.stdout,'stderr':build.stderr,'library_sha256':before},indent=2)+'\n')
        build.check_returncode()
        runner.OUT = OUT
        runner.run('cpu-calibration', ['numactl','--physcpubind=0-23','--interleave=0,1',str(binary)])
        assert hashlib.sha256((lib/'libggml-cpu.so').read_bytes()).hexdigest() == before
        print('CPU calibration complete',flush=True)


if __name__ == '__main__':
    main()
