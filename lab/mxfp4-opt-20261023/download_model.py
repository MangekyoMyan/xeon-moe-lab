#!/usr/bin/env python3
"""Fetch a pinned public MXFP4 model and verify its published SHA256."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

LAB = Path(__file__).resolve().parent.parent
OUT = LAB / 'results/mxfp4-opt-20261023'
DEST = LAB / 'models/ggml-org--gpt-oss-20b-GGUF'
REPO = 'ggml-org/gpt-oss-20b-GGUF'
NAME = 'gpt-oss-20b-MXFP4.gguf'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    DEST.mkdir(exist_ok=True)
    info = get_json(f'https://huggingface.co/api/models/{REPO}')
    revision = info['sha']
    files = get_json(f'https://huggingface.co/api/models/{REPO}/tree/{revision}')
    entry = next(x for x in files if x['path'] == NAME)
    expected_size, expected_sha = entry['size'], entry['lfs']['oid']
    url = f'https://huggingface.co/{REPO}/resolve/{revision}/{NAME}'
    manifest = {'repository': REPO, 'revision': revision, 'file': NAME, 'bytes': expected_size, 'sha256': expected_sha, 'url': url, 'complete': False}
    record = OUT / 'model-download.json'
    record.write_text(json.dumps(manifest, indent=2) + '\n')
    final, partial = DEST / NAME, DEST / (NAME + '.partial')
    if final.exists():
        raise RuntimeError('Destination already exists; refusing to replace it')
    offset = partial.stat().st_size if partial.exists() else 0
    assert shutil.disk_usage(DEST).free > expected_size - offset + 30 * 1024 ** 3
    digest = hashlib.sha256()
    if offset:
        with partial.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
    begin = time.monotonic()
    last = begin
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset:
                    assert response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                with partial.open('ab') as handle:
                    while block := response.read(4 * 1024 * 1024):
                        assert offset + len(block) <= expected_size
                        handle.write(block)
                        digest.update(block)
                        offset += len(block)
                        now = time.monotonic()
                        if now - last >= 30:
                            print(json.dumps({'downloaded_GB': round(offset / 1e9, 3), 'total_GB': round(expected_size / 1e9, 3), 'elapsed_s': round(now - begin)}), flush=True)
                            last = now
            if offset == expected_size:
                break
        except (OSError, TimeoutError) as error:
            print(json.dumps({'retry': attempt + 1, 'error_type': type(error).__name__, 'bytes': offset}), flush=True)
            if attempt == 5:
                raise
    assert offset == expected_size and digest.hexdigest() == expected_sha
    partial.rename(final)
    manifest.update(complete=True, observed_sha256=digest.hexdigest(), elapsed_s=time.monotonic()-begin, path=str(final))
    record.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'sha256': digest.hexdigest(), 'bytes': offset}), flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        owner = LAB.stat()
        if os.geteuid() == 0:
            for path in [DEST, *DEST.glob('*'), OUT / 'model-download.json']:
                if path.exists():
                    os.chown(path, owner.st_uid, owner.st_gid)
