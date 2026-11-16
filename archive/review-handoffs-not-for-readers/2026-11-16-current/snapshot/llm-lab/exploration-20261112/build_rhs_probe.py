from pathlib import Path
import hashlib
import json
import subprocess

R = Path(__file__).resolve().parent
LAB = R.parent
OUT = R / 'rhs-prepack'
(OUT/'compact-kernel.inc').write_bytes((LAB/'iq-cache-next-20261103/compact-kernel.inc').read_bytes())
source = (OUT/'probe.cpp').read_text()
body = source[source.index('__attribute__((noinline)) static float prepared_dot'):source.index('#include "variants.inc"')]
assert body.count('for (int g = 0; g < 8; ++g) {') == 1
variants = []
for unroll in [1,2,4]:
    code = body.replace('prepared_dot(', 'prepared_dot_u'+str(unroll)+'(')
    code = code.replace('for (int g = 0; g < 8; ++g) {', '#pragma GCC unroll '+str(unroll)+'\n        for (int g = 0; g < 8; ++g) {')
    variants.append(code)
(OUT/'variants.inc').write_text('\n'.join(variants))
command = json.loads((LAB/'optimization-20261112/cpu-bias/BUILD.json').read_text())['command']
command = [str(OUT/'probe.cpp') if x.endswith('/cpu-bias/test-bias.cpp') else
           str(OUT/'probe') if x.endswith('/cpu-bias/test-bias') else x for x in command]
subprocess.run(command,check=True,timeout=180)
(OUT/'BUILD.json').write_text(json.dumps({'command':command,
    'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir() if f.is_file()}},indent=2)+'\n')
subprocess.run([str(OUT/'probe')],check=True,timeout=30)
