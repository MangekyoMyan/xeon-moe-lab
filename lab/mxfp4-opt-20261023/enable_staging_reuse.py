#!/usr/bin/env python3
"""Reuse same-size synchronous CPU_REPACK staging memory, with bounded lifetime."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import subprocess

LAB = Path(__file__).resolve().parent.parent
ROOT = LAB / 'review-mxfp4-final-20261023'


def main():
    path = ROOT / 'source/src/llama-model-loader.cpp'
    original = path.read_text()
    assert 'GGML_CPU_REPACK_REUSE_BUFFER' not in original
    code = original.replace('#include <cstring>', '#include <cstring>\n#include <cstdlib>')
    start = code.index('bool llama_model_loader::load_all_data(')
    section = code[start:]
    anchor = '    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;'
    assert section.count(anchor) == 1
    section = section.replace(anchor, anchor + '''
    std::vector<no_init<uint8_t>> repack_staging;
    const char * reuse_env = std::getenv("GGML_CPU_REPACK_REUSE_BUFFER");
    const bool allow_repack_reuse = reuse_env != nullptr && std::strcmp(reuse_env, "1") == 0;
''')
    anchor = '        const bool from_mapping = use_mmap || lazy.has(cur);'
    assert section.count(anchor) == 1
    section = section.replace(anchor, anchor + '''
        const bool reuse_repack_staging = allow_repack_reuse && !from_mapping && !upload_backend &&
            cur->type == GGML_TYPE_MXFP4 && cur->buffer &&
            std::strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)), "CPU_REPACK") == 0;
        if (!reuse_repack_staging || repack_staging.size() != n_size) {
            std::vector<no_init<uint8_t>>().swap(repack_staging);
        }
''')
    anchor = '''                    // scoped to one tensor so only one staging buffer is alive at a time
                    std::vector<no_init<uint8_t>> read_buf(n_size);'''
    replacement = '''                    // Keep one CPU repack staging allocation while tensor sizes match.
                    std::vector<no_init<uint8_t>> local_read_buf;
                    auto & read_buf = reuse_repack_staging ? repack_staging : local_read_buf;
                    read_buf.resize(n_size);'''
    assert section.count(anchor) == 1
    section = section.replace(anchor, replacement)
    path.write_text(code[:start] + section)
    (ROOT / 'staging-reuse.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), path.read_text().splitlines(True), fromfile='a/src/llama-model-loader.cpp', tofile='b/src/llama-model-loader.cpp')))
    build = json.loads((ROOT / 'build-commands.json').read_text())[1]
    with (ROOT / 'build-staging-reuse.log').open('w') as log:
        subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Staging reuse build complete', flush=True)


if __name__ == '__main__':
    try:
        with (LAB / 'results/glm-air-q8-20261020/.experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    finally:
        if os.geteuid() == 0:
            owner = LAB.stat()
            for path in [ROOT, *ROOT.rglob('*')]:
                os.chown(path, owner.st_uid, owner.st_gid, follow_symlinks=False)
