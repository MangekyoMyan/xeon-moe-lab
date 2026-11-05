#!/usr/bin/env python3
"""Move four cached 1 MiB samples without privileges, then restore every page."""
import collections
import ctypes
import hashlib
import json
import mmap
import os
import time

from page_placement import LAB, LIBC, PAGE, move

def main():
    assert os.geteuid() == LAB.stat().st_uid and os.geteuid() != 0
    with open('/proc/self/status') as status:
        capabilities = next(line.split(':', 1)[1].strip() for line in status if line.startswith('CapEff:'))
    assert int(capabilities, 16) == 0
    path = LAB / 'models/ggml-org--GLM-4.5-Air-GGUF/GLM-4.5-Air-Q8_0-00002-of-00002.gguf'
    before_stat = path.stat()
    report = {'uid': os.geteuid(), 'effective_capabilities': capabilities, 'flags': 0, 'samples': []}
    started = time.monotonic()
    with path.open('rb') as stream:
        mapping = mmap.mmap(stream.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        anchor = ctypes.c_char.from_buffer(mapping)
        base = ctypes.addressof(anchor)
        for offset in [1 << 30, 32 << 30, 64 << 30, 96 << 30]:
            size = 1 << 20
            resident = (ctypes.c_ubyte * (size // PAGE))()
            assert LIBC.mincore(base + offset, size, resident) == 0 and all(x & 1 for x in resident)
            expected = hashlib.sha256(mapping[offset:offset + size]).hexdigest()
            addresses = [base + offset + i * PAGE for i in range(size // PAGE)]
            original = move(addresses)
            assert all(node in [0, 1] for node in original)
            target = [1 - node for node in original]
            try:
                moved = move(addresses, target, flags=0)
                assert moved == target
                assert hashlib.sha256(mapping[offset:offset + size]).hexdigest() == expected
            finally:
                restored = move(addresses, original, flags=0)
                assert restored == original
            assert hashlib.sha256(mapping[offset:offset + size]).hexdigest() == expected
            report['samples'].append({'offset': offset, 'bytes': size, 'original_nodes': dict(collections.Counter(original)),
                                      'moved_and_restored': True, 'sha256': expected})
        del anchor
        mapping.close()
    after_stat = path.stat()
    assert (before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ino) == (after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ino)
    report.update(passed=True, elapsed_s=time.monotonic() - started, file_stat_unchanged=True)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
