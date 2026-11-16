#!/usr/bin/env python3
"""Extract three exact IQ3_S symbols from GNU objdump -d -C (AT&T) text.

This reads assembly only: it never builds or executes a binary. Counts are static
mnemonic occurrences; vector stack operands are not dynamic spill/cycle counts.
Unknown signatures, missing functions, and duplicate/clone candidates fail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


SYMBOLS = {
    'current_iq3s': 'float iq_compact_dot<false>(int, iq_compact_block const*, block_q8_K const*)',
    'offset4_fast': 'offset4_fast_dot(int, offset4_iq3s_block const*, block_q8_K const*, offset4_rhs_block const*)',
    'offset4_compat': 'offset4_compat_dot(int, offset4_iq3s_block const*, block_q8_K const*)',
}
FAMILIES = {
    'current_iq3s': re.compile(r'(?:^|\s)iq_compact_dot<false>\('),
    'offset4_fast': re.compile(r'(?:^|\s)offset4_fast_dot\('),
    'offset4_compat': re.compile(r'(?:^|\s)offset4_compat_dot\('),
}
OPCODES = ('vpshufb', 'vpsignb', 'vpabsb', 'vpmaddubsw', 'vpaddw', 'vpsubw')
HEADER = re.compile(r'(?m)^[0-9a-fA-F]+ <(?P<symbol>.+)>:$')
# Require GNU objdump's address, byte column, and separate mnemonic column.
INSTRUCTION = re.compile(r'^\s*[0-9a-fA-F]+:\s+(?:[0-9a-fA-F]{2}[ \t]+)+(?P<opcode>[a-z][a-z0-9.]*)\b(?P<operands>.*)$')
VECTOR_REGISTER = re.compile(r'%[xyz]mm[0-9]+\b')
STACK_MEMORY = re.compile(r'\([^)]*%(?:r|e)(?:sp|bp)(?:[,)])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(assembly):
    headers = list(HEADER.finditer(assembly))
    functions, extracted = {}, {}
    for key, exact_symbol in SYMBOLS.items():
        candidates = [(i, m) for i, m in enumerate(headers)
                      if FAMILIES[key].search(m.group('symbol'))]
        if len(candidates) != 1:
            raise ValueError(f'{key}: expected exactly one function header, found {len(candidates)}')
        i, header = candidates[0]
        if header.group('symbol') != exact_symbol:
            raise ValueError(f'{key}: unsupported signature/clone: {header.group("symbol")!r}')
        end = headers[i+1].start() if i+1 < len(headers) else len(assembly)
        block = assembly[header.start():end]
        counts = dict.fromkeys(OPCODES, 0)
        stack_lines = []
        parsed = 0
        for line in block.splitlines()[1:]:
            instruction = INSTRUCTION.match(line)
            if instruction is None:
                continue  # Blank lines and byte-only continuation lines.
            parsed += 1
            opcode = instruction.group('opcode')
            operands = instruction.group('operands').split('#', 1)[0]
            if opcode in counts:
                counts[opcode] += 1
            if VECTOR_REGISTER.search(operands) and STACK_MEMORY.search(operands):
                stack_lines.append(line.strip())
        if parsed == 0 or not VECTOR_REGISTER.search(block):
            raise ValueError(f'{key}: no recognized AT&T vector instruction body')
        functions[key] = {
            'header': header.group(0),
            'static_opcode_counts': counts,
            'vector_stack_operand_instructions': stack_lines,
            'assembly_sha256': sha(block.encode('utf-8')),
        }
        extracted[key] = block.encode('utf-8')
    return functions, extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory; never overwritten')
    parser.add_argument('--expected-summary', type=Path,
                        help='optional existing summary; require exact function records and hashes')
    args = parser.parse_args()
    assembly_path, out = args.assembly.resolve(), args.output.resolve()
    source = assembly_path.read_bytes()
    functions, extracted = analyze(source.decode('utf-8'))
    receipt = {
        'scope': 'Read-only static assembly extraction; no build, ELF execution, or performance test.',
        'assembly_path': str(assembly_path), 'assembly_sha256': sha(source),
        'analyzer_sha256': sha(Path(__file__).read_bytes()),
        'exact_single_header_per_function': True,
        'performance_measured': False,
    }
    if args.expected_summary is not None:
        expected_path = args.expected_summary.resolve()
        expected_bytes = expected_path.read_bytes()
        expected = json.loads(expected_bytes)['functions']
        checks = {key: functions[key] == expected.get(key) for key in SYMBOLS}
        receipt['expected_summary_path'] = str(expected_path)
        receipt['expected_summary_sha256'] = sha(expected_bytes)
        receipt['exact_function_record_matches'] = checks
        if set(expected) != set(SYMBOLS) or not all(checks.values()):
            raise ValueError(f'reproduced function records differ from expected summary: {checks}')
    summary = {
        'scope': 'Static opcode and vector stack operand counts from supplied GNU objdump AT&T text; not cycles, dynamic counts, or performance prediction.',
        'functions': functions, 'performance_measured': False,
    }
    out.mkdir(parents=True, exist_ok=False)
    for key, block in extracted.items():
        (out/(key+'.asm')).write_bytes(block)
    (out/'ASSEMBLY_SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    receipt['passed'] = True
    receipt['outputs'] = {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
    (out/'REPRODUCTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'output': str(out),
                      'function_sha256': {key: row['assembly_sha256'] for key, row in functions.items()},
                      'expected_summary_checked': args.expected_summary is not None}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        raise SystemExit(1)
