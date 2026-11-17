"""Build the isolated caller only; do not relink or modify engine libraries."""
from pathlib import Path
import hashlib
import json
import subprocess

R = Path(__file__).resolve().parent
LAB = R.parent.parent
S = LAB/'deepseek41-bringup-20261101/source'
B = LAB/'deepseek41-bringup-20261101/build/bin'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (R/'BUILD.json').exists(), 'Preserve prior build receipt before rebuilding'
    cmd = ['g++','-std=c++17','-O2','-g','-march=ivybridge','-mtune=ivybridge',
           '-mno-avx2','-mno-fma','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra',
           '-fopenmp','-pthread']
    for inc in [R,R/'route37',S/'include',S/'ggml/include',S/'ggml/src',S/'common',S/'src',S/'tools/mtmd',S/'vendor/nlohmann']:
        cmd += ['-I'+str(inc)]
    cmd += [str(R/'driver.cpp'),'-L'+str(R/'runtime'),'-L'+str(B),
            '-Wl,-rpath,'+str(R/'runtime')+':'+str(B),'-Wl,--no-as-needed',
            '-lllama','-lggml-cpu','-lggml-base','-lggml','-lllama-common','-lmtmd','-ldl',
            '-Wl,--no-undefined','-o',str(R/'driver')]
    with (R/'BUILD.stdout.log').open('w') as out, (R/'BUILD.stderr.log').open('w') as err:
        result = subprocess.run(cmd,stdout=out,stderr=err,timeout=180)
    receipt = dict(command=cmd,returncode=result.returncode,
        source_sha256={str(p.relative_to(R)):sha(p) for p in [R/'driver.cpp',R/'route_client.h',R/'call_control.h',R/'route37/route37.h',R/'CASES.json',R/'DESIGN.json',R/'PILOT_DESIGN.json',R/'BASELINE_PIN.json',R/'capture_io.h',R/'bounded_moe_capture_proposal.h',R/'bind_parent_cells_proposal.h']},
        engine_libraries={arm:{p.name:dict(path=str(p.resolve()),sha256=sha(p)) for p in (R/'runtime').iterdir() if p.name.endswith(('.0.8.0','.0.28.0'))} for arm in ['diagnostic']})
    if not result.returncode: receipt['driver_sha256']=sha(R/'driver')
    (R/'BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'returncode':result.returncode,'receipt':str(R/'BUILD.json')}))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
