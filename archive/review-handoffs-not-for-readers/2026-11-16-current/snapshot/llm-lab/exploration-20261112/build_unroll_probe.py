from pathlib import Path
import hashlib
import json
import subprocess

R = Path(__file__).resolve().parent
LAB = R.parent
OUT = R / 'compact-unroll'
OUT.mkdir(exist_ok=True)
OLD = LAB / 'iq-cache-next-20261103'
for name in ['compact-kernel.inc', 'iq-small-kernels.inc']:
    (OUT / name).write_bytes((OLD / name).read_bytes())
variants = []
for name, marker, function in [
    ('compact-kernel.inc', '__attribute__((noinline)) static float compact_dot', 'compact_dot'),
    ('iq-small-kernels.inc', 'template<bool two> __attribute__((noinline)) static float iq_compact_dot', 'iq_compact_dot'),
]:
    body = (OLD / name).read_text().split(marker, 1)[1]
    body = marker + body
    assert body.count('for(int g=0;g<8;++g) {') == 1
    for unroll in [1, 2, 4, 8]:
        code = body.replace(function + '(', function + '_u' + str(unroll) + '(')
        code = code.replace('for(int g=0;g<8;++g) {', '#pragma GCC unroll ' + str(unroll) + '\n        for(int g=0;g<8;++g) {')
        variants.append(code)
(OUT / 'variants.inc').write_text('\n'.join(variants))
command = json.loads((LAB / 'optimization-20261112/cpu-bias/BUILD.json').read_text())['command']
command = [str(OUT / 'probe.cpp') if x.endswith('/cpu-bias/test-bias.cpp') else
           str(OUT / 'probe') if x.endswith('/cpu-bias/test-bias') else x for x in command]
subprocess.run(command, check=True, timeout=180)
(OUT / 'BUILD.json').write_text(json.dumps({'command': command,
    'sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file()}}, indent=2) + '\n')
subprocess.run([str(OUT / 'probe')], check=True, timeout=30)
