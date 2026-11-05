#!/usr/bin/env python3
"""Test bounded prefetch to recover streaming bandwidth with shared rows."""
import hashlib
import json
import shlex
import subprocess
from prepare_rows import BASE, SOURCE, ROOT, kernel


def main():
    entry = next(x for x in json.loads((BASE / 'build/compile_commands.json').read_text()) if x['file'] == str(SOURCE))
    flags = shlex.split(entry['command'])
    flags = ['-march=ivybridge' if x == '-march=native' else x for x in flags[:flags.index('-o')]]
    record = []
    for distance in (8, 16, 32):
        name = f'rows4-pf{distance}'
        function = kernel(4, True)
        anchor = '        for (int r = 0; r < 4; ++r) {\n'
        assert function.count(anchor) == 1
        function = function.replace(anchor, anchor + f'            if (ib + {distance} < nb) {{\n                _mm_prefetch((const char *) &x[r][ib + {distance}], _MM_HINT_T0);\n            }}\n')
        path = ROOT / (name + '.c')
        path.write_text((ROOT / 'narrow.c').read_text() + '\n' + function)
        (ROOT / (name + '-kernel.c')).write_text(function)
        command = flags + ['-iquote', str(SOURCE.parent), '-shared', '-Wl,-Bsymbolic', str(path), '-L' + str(BASE / 'build/bin'), '-Wl,-rpath,' + str(BASE / 'build/bin'), '-lggml-cpu', '-lggml-base', '-lm', '-o', str(ROOT / (name + '.so'))]
        with (ROOT / (name + '.compile.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record.append({'name': name, 'distance_blocks': distance, 'command': command, 'sha256': hashlib.sha256((ROOT / (name + '.so')).read_bytes()).hexdigest()})
        print('compiled ' + name, flush=True)
    (ROOT / 'prefetch-build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
