#!/usr/bin/env python3
"""Index cached GGUF metadata and expose existing blobs through hard links."""

import datetime
import json
import os
from pathlib import Path
import struct

CACHE = Path('/root/.cache/huggingface/hub')
LAB = Path('/home/codex/projects/llm-lab')
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def unpack(stream, fmt):
    size = struct.calcsize('<' + fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated GGUF metadata')
    return struct.unpack('<' + fmt, data)[0]


def string(stream, keep=True):
    size = unpack(stream, 'Q')
    if size > os.fstat(stream.fileno()).st_size - stream.tell():
        raise ValueError('String exceeds file size')
    if keep:
        if size > 16 * 1024 * 1024:
            raise ValueError('Metadata string too large')
        return stream.read(size).decode('utf-8')
    stream.seek(size, 1)


def value(stream, kind, keep=True):
    if kind in SCALARS:
        return unpack(stream, SCALARS[kind])
    if kind == 8:
        return string(stream, keep)
    if kind == 9:
        subtype, count = unpack(stream, 'I'), unpack(stream, 'Q')
        if count > 10000000:
            raise ValueError('Metadata array too large')
        if subtype in SCALARS:
            size = count * struct.calcsize('<' + SCALARS[subtype])
            if size > os.fstat(stream.fileno()).st_size - stream.tell():
                raise ValueError('Array exceeds file size')
            stream.seek(size, 1)
        elif subtype == 8:
            for _ in range(count):
                string(stream, False)
        else:
            raise ValueError('Unsupported metadata array type')
        return {'array_type': subtype, 'count': count}
    raise ValueError(f'Unknown GGUF metadata type: {kind}')


def metadata(path):
    with path.open('rb') as stream:
        if stream.read(4) != b'GGUF':
            raise ValueError('Not a little-endian GGUF file')
        version = unpack(stream, 'I')
        if version not in (2, 3):
            raise ValueError(f'Unsupported GGUF version: {version}')
        tensors, count = unpack(stream, 'Q'), unpack(stream, 'Q')
        if count > 100000:
            raise ValueError('Too many metadata entries')
        fields = {}
        for _ in range(count):
            key = string(stream)
            kind = unpack(stream, 'I')
            keep = not key.startswith('tokenizer.')
            item = value(stream, kind, keep)
            if keep and kind != 9:
                fields[key] = item
        return {'version': version, 'tensor_count': tensors, 'fields': fields}


def main():
    owner = LAB.stat()
    records = []
    for source in sorted(CACHE.glob('models--*/snapshots/**/*.gguf')):
        relative = source.relative_to(CACHE)
        repo, _, revision, *filename = relative.parts
        blob = source.resolve(strict=True)
        blob.relative_to(CACHE / repo / 'blobs')
        info = blob.stat()
        record = {'repository': repo.removeprefix('models--').replace('--', '/'),
                  'revision': revision, 'snapshot_path': str(source),
                  'blob_path': str(blob), 'size_bytes': info.st_size,
                  'size_gib': round(info.st_size / 2**30, 3),
                  'validation': 'GGUF header and metadata only; weight data not hashed'}
        record['gguf'] = metadata(blob)
        target = LAB / 'models' / repo.removeprefix('models--') / Path(*filename)
        if not info.st_mode & 0o004:
            raise PermissionError(f'Blob is not world-readable: {blob}')
        missing = []
        parent = target.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir()
            os.chown(directory, owner.st_uid, owner.st_gid)
        if target.exists():
            if not target.samefile(blob):
                raise FileExistsError(target)
        else:
            os.link(blob, target)
        record['local_path'] = str(target)
        record['storage'] = 'hard link; shares the original blob inode and disk blocks'
        records.append(record)
    report = {'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'cache': str(CACHE), 'files': records,
              'deletion_note': 'Deleting a cache entry alone will not free a blob while a local hard link remains.'}
    output = LAB / 'model-inventory.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    os.chown(output, owner.st_uid, owner.st_gid)
    print(json.dumps({'inventory': str(output), 'files': [
        {key: item[key] for key in ['repository', 'size_gib', 'local_path', 'gguf']}
        for item in records]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
