#!/usr/bin/env python3
"""Inspect or migrate cached GGUF pages without writing the model file."""
import argparse
import collections
import ctypes
import fcntl
import json
import mmap
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
OUT = LAB / 'results' / ROOT.name
HEADERS = LAB / 'results/glm-air-q8-preparation-20261020/header-inspection.json'
MODELS = LAB / 'models/ggml-org--GLM-4.5-Air-GGUF'
PAGE = os.sysconf('SC_PAGE_SIZE')
LIB = ctypes.CDLL('libnuma.so.1', use_errno=True)
LIB.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.numa_move_pages.restype = ctypes.c_int
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
LIBC.mincore.restype = ctypes.c_int

def move(addresses, targets=None, flags=4):
    n = len(addresses)
    pages = (ctypes.c_void_p * n)(*addresses)
    nodes = None if targets is None else (ctypes.c_int * n)(*targets)
    status = (ctypes.c_int * n)()
    result = LIB.numa_move_pages(0, n, pages, nodes, status, 0 if nodes is None else flags)
    if result < 0:
        raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
    return list(status)

def selftest(flags=4):
    memory = mmap.mmap(-1, PAGE * 128, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    memory[:] = bytes([91]) * len(memory)
    anchor = ctypes.c_char.from_buffer(memory)
    base = ctypes.addressof(anchor)
    addresses = [base + p * PAGE for p in range(128)]
    before = move(addresses)
    target = [p % 2 for p in range(128)]
    after = move(addresses, target, flags)
    assert after == target and memory[:] == bytes([91]) * len(memory)
    print(json.dumps({'selftest': True, 'flags': flags, 'before': dict(collections.Counter(before)), 'after': dict(collections.Counter(after))}))
    del anchor
    memory.close()
    with tempfile.TemporaryFile(dir=OUT) as stream:
        content = bytes([73]) * (PAGE * 4096)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        memory = mmap.mmap(stream.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        assert memory[:] == content
        anchor = ctypes.c_char.from_buffer(memory)
        base = ctypes.addressof(anchor)
        addresses = [base + p * PAGE for p in range(4096)]
        target = [p % 2 for p in range(4096)]
        after = move(addresses, target, flags)
        assert after == target and memory[:] == content
        stream.seek(0)
        assert stream.read() == content
        print(json.dumps({'file_backed_selftest': True, 'flags': flags, 'after': dict(collections.Counter(after)), 'bytes': len(content)}))
        del anchor
        memory.close()

def process(mode, label):
    headers = json.loads(HEADERS.read_text())['parts']
    report = {'mode': mode, 'label': label, 'start_unix': time.time(), 'parts': {},
              'writes_to_file': False, 'page_size': PAGE}
    for header_name, info in headers.items():
        path = MODELS / header_name.removesuffix('.header')
        stat = path.stat()
        counts, target_counts = collections.Counter(), collections.Counter()
        with path.open('rb') as stream:
            # Writable private address space only to expose its pointer to ctypes.
            # No stores, write syscall, or dirtying of model pages is performed.
            memory = mmap.mmap(stream.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
            anchor = ctypes.c_char.from_buffer(memory)
            base = ctypes.addressof(anchor)
            resident = (ctypes.c_ubyte * ((len(memory) + PAGE - 1) // PAGE))()
            if LIBC.mincore(base, len(memory), resident):
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
            resident_fraction = sum(bool(x & 1) for x in resident) / len(resident)
            if mode != 'inspect' and resident_fraction < 0.9999:
                raise RuntimeError(f'Warm the entire model first: resident fraction={resident_fraction}')
            addresses, targets = [], []
            started = time.monotonic()
            previous_progress = started
            if mode == 'inspect':
                offsets = range(0, len(memory), 1 << 20)
                for offset in offsets:
                    if not resident[offset // PAGE] & 1:
                        counts[-2] += 1
                        continue
                    _ = memory[offset]
                    addresses.append(base + offset)
                    if len(addresses) == 4096:
                        counts.update(move(addresses))
                        addresses.clear()
                if addresses:
                    counts.update(move(addresses))
            else:
                # Touch one byte per sampled page before queries. mmap PTEs must
                # be present; the full file remains resident after model warm-up.
                last_page = -1
                for tensor in sorted(info['tensors'], key=lambda t: t['offset']):
                    start = info['data_offset'] + tensor['offset']
                    end = start + tensor['bytes']
                    if tensor['type'] != 'Q8_0':
                        continue
                    dims = tensor['dims']
                    expert_bytes = dims[0] * dims[1] // 32 * 34
                    for offset in range((start + PAGE - 1) // PAGE * PAGE, end // PAGE * PAGE, PAGE):
                        assert offset > last_page
                        last_page = offset
                        if mode == 'interleave':
                            target = (offset // PAGE) % 2
                        elif mode in ['stripe2m', 'stripe16m', 'stripe64m']:
                            stripe = {'stripe2m': 2, 'stripe16m': 16, 'stripe64m': 64}[mode] << 20
                            target = (offset // stripe) % 2
                        else:
                            within_expert = (offset - start) % expert_bytes
                            target = int(within_expert >= expert_bytes // 2)
                        _ = memory[offset]
                        addresses.append(base + offset)
                        targets.append(target)
                        if len(addresses) == 4096:
                            status = move(addresses, targets)
                            counts.update(status)
                            target_counts.update(targets)
                            if any(s < 0 for s in status):
                                raise RuntimeError(f'Migration status error: {dict(collections.Counter(status))}')
                            if status != targets:
                                raise RuntimeError('Page placement differs from requested node')
                            addresses.clear(); targets.clear()
                        now = time.monotonic()
                        if now - previous_progress >= 30:
                            print(json.dumps({'part': path.name, 'seconds': now - started,
                                              'processed_GB': sum(counts.values()) * PAGE / 1e9}), flush=True)
                            previous_progress = now
                if addresses:
                    status = move(addresses, targets)
                    counts.update(status); target_counts.update(targets)
                    assert status == targets
            del anchor
            memory.close()
        after = path.stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
        report['parts'][path.name] = {'counts': dict(counts), 'target_counts': dict(target_counts),
                                     'resident_fraction': resident_fraction,
                                     'elapsed_s': time.monotonic() - started, 'stat_unchanged': True}
        (OUT / (label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    report['elapsed_s'] = time.time() - report['start_unix']
    (OUT / (label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['selftest', 'inspect', 'interleave', 'rowlocal', 'stripe2m', 'stripe16m', 'stripe64m'])
    parser.add_argument('--label', default='page-placement')
    parser.add_argument('--unprivileged', action='store_true', help='Self-test without MPOL_MF_MOVE_ALL')
    args = parser.parse_args()
    if args.mode == 'selftest':
        return selftest(0 if args.unprivileged else 4)
    with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process(args.mode, args.label)

if __name__ == '__main__':
    main()
