#!/usr/bin/env python3
"""Prepare bounded AVX Q8 prefetch variants without changing the active source."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

lab = Path(__file__).resolve().parent
out = lab / 'q8-prefetch-20261019'
out.mkdir(exist_ok=False)
source = lab / 'llama-private/ggml/src/ggml-cpu/arch/x86/quants.c'
original = source.read_text()
start = original.index('void ggml_vec_dot_q8_0_q8_0(')
end = original.index('\n#if defined(__AVX__) && !defined(__AVX2__)', start)
fn = original[start:end]
anchor = '    for (; ib + 1 < nb; ib += 2) {\n'
assert fn.count(anchor) == 1
entry = next(x for x in json.loads((lab / 'build-avx-deep/compile_commands.json').read_text())
             if x['file'] == str(source))
flags = shlex.split(entry['command'])
flags = flags[:flags.index('-o')]
commands = []
for distance in (0, 8, 16, 32):
    addition = (f'        if (ib + {distance} < nb) {{\n'
                f'            _mm_prefetch((const char *) &x[ib + {distance}], _MM_HINT_T0);\n'
                '        }\n') if distance else ''
    text = original[:start] + fn.replace(anchor, anchor + addition) + original[end:]
    path = out / f'prefetch-{distance}.c'
    path.write_text(text)
    (out / f'prefetch-{distance}.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), text.splitlines(True),
        fromfile='a/ggml/src/ggml-cpu/arch/x86/quants.c',
        tofile='b/ggml/src/ggml-cpu/arch/x86/quants.c')))
    command = flags + ['-iquote', str(source.parent), '-shared', '-Wl,-Bsymbolic',
        str(path), '-L' + str(lab / 'build-avx-deep/bin'),
        '-Wl,-rpath,' + str(lab / 'build-avx-deep/bin'), '-lggml-cpu', '-lggml-base',
        '-lm', '-o', str(out / f'prefetch-{distance}.so')]
    commands.append(command)
(out / 'prepare.json').write_text(json.dumps({'baseline_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'commands': commands, 'adopted': False}, indent=2) + '\n')
for i, command in enumerate(commands):
    with (out / f'compile-{i}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'compiled': command[-1]}), flush=True)
