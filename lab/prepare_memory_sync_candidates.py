#!/usr/bin/env python3
"""Prepare separate private Q8 and barrier experiments from the frozen baseline."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent
baseline = lab / 'llama-private'
variants = {
    'q8-prefetch8': Path('ggml/src/ggml-cpu/arch/x86/quants.c'),
    'native-barrier': Path('ggml/src/ggml-cpu/ggml-cpu.c'),
}
for name, relative in variants.items():
    out = lab / ('review-' + name + '-20261019')
    out.mkdir(exist_ok=False)
    shutil.copytree(baseline, out / 'source')
    before = (baseline / relative).read_text()
    if name == 'q8-prefetch8':
        after = (lab / 'q8-prefetch-20261019/prefetch-8.c').read_text()
    else:
        start = before.index('void ggml_barrier(')
        end = before.index('\nvoid ggml_threadpool_chunk_set(', start)
        function = before[start:end]
        anchor = '#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else\n'
        assert function.count(anchor) == 1
        # Use the existing C11 barrier inside the existing OpenMP worker team.
        function = function.replace(anchor, '')
        assert function.endswith('#endif\n}\n')
        function = function[:-len('#endif\n}\n')] + '}\n'
        after = before[:start] + function + before[end:]
    (out / 'source' / relative).write_text(after)
    (out / (name + '.patch')).write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile='a/' + str(relative), tofile='b/' + str(relative))))
    (out / 'prepare.json').write_text(json.dumps({'baseline': str(baseline),
        'files': {str(relative): {'before': hashlib.sha256(before.encode()).hexdigest(),
                                'after': hashlib.sha256(after.encode()).hexdigest()}},
        'adopted': False, 'scope': 'Private CPU-only experiment; no changes to active source or launcher'}, indent=2) + '\n')
    print(out)
