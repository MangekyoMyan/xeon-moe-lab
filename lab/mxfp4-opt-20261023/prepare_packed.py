#!/usr/bin/env python3
"""Keep MXFP4 bytes unchanged in count, interleaving four rows by whole blocks."""
import hashlib
import json
import shlex
import subprocess
from prepare_rows import BASE, SOURCE, ROOT, kernel


def packed_kernel(distance=0):
    code = kernel(4, True)
    code = code.replace('    const block_mxfp4 * x[4];', '    const uint8_t * packed = vx;')
    code = code.replace('        x[r] = (const block_mxfp4 *) ((const char *) vx + r * bx);\n', '')
    code = code.replace('x[r][ib + 1].qs', '(packed + (ib + 1) * 68 + 4 + r * 16)')
    code = code.replace('x[r][ib].qs', '(packed + ib * 68 + 4 + r * 16)')
    code = code.replace('x[r][ib + 1].e', 'packed[(ib + 1) * 68 + r]')
    code = code.replace('x[r][ib].e', 'packed[ib * 68 + r]')
    if distance:
        anchor = '    for (; ib + 1 < nb; ib += 2) {\n'
        code = code.replace(anchor, anchor + f'        if (ib + {distance} < nb) {{\n            _mm_prefetch((const char *) (packed + (ib + {distance}) * 68), _MM_HINT_T0);\n        }}\n')
    code = code.replace('ib * 68', '(size_t) ib * 68')
    code = code.replace('(ib + 1) * 68', '(size_t) (ib + 1) * 68')
    if distance:
        code = code.replace(f'(ib + {distance}) * 68', f'(size_t) (ib + {distance}) * 68')
    return code


def main():
    entry = next(x for x in json.loads((BASE / 'build/compile_commands.json').read_text()) if x['file'] == str(SOURCE))
    flags = shlex.split(entry['command'])
    flags = ['-march=ivybridge' if x == '-march=native' else x for x in flags[:flags.index('-o')]]
    record = []
    for distance in (0, 8, 16):
        name = 'rows4-packed' + (f'-pf{distance}' if distance else '')
        function = packed_kernel(distance)
        path = ROOT / (name + '.c')
        path.write_text((ROOT / 'narrow.c').read_text() + '\n' + function)
        (ROOT / (name + '-kernel.c')).write_text(function)
        command = flags + ['-iquote', str(SOURCE.parent), '-shared', '-Wl,-Bsymbolic', str(path), '-L' + str(BASE / 'build/bin'), '-Wl,-rpath,' + str(BASE / 'build/bin'), '-lggml-cpu', '-lggml-base', '-lm', '-o', str(ROOT / (name + '.so'))]
        with (ROOT / (name + '.compile.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record.append({'name': name, 'distance_blocks': distance, 'command': command, 'sha256': hashlib.sha256((ROOT / (name + '.so')).read_bytes()).hexdigest()})
        print('compiled ' + name, flush=True)
    (ROOT / 'packed-build.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
