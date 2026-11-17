#!/usr/bin/env python3
"""Create an optional diagnostic B-control patch without editing the repository.

Only reads the pinned source and writes this proposal. No supplied script,
engine, model, or GPU is executed. Applying this patch is NOT sufficient to run
the current driver as B: its arm labels and expected call counts need the
explicit changes described in DORMANT_BANK_CONTROL_JA.txt.
"""
from pathlib import Path
import difflib
import hashlib
import json
import subprocess
import sys

PIN = "d4d50b651ff7754ee2189228762468fc02971bfa"
REL = "archive/review-handoffs-not-for-readers/2026-11-17-parallel-pilot/integrated/hot-backend.cpp"
SOURCE_SHA = "270b7f6f4398f5d1978026bebc18e129614956d028414a98daf76f91374a881c"


def main():
    repo, out = (Path(x).resolve() for x in sys.argv[1:3])
    data = (repo/REL).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise SystemExit("Source differs from the reviewed pin; do not apply by guess.")
    old = data.decode()
    new = old
    changes = [
        (
            'struct State{\n',
            'bool original_graph_trial_flag(){\n'
            '    const char * value=std::getenv("GGML_XEON_HOT_TRIAL_KEEP_ORIGINAL_GRAPH");\n'
            '    require(!value||!std::strcmp(value,"0")||!std::strcmp(value,"1"),"original graph trial flag must be 0 or 1");\n'
            '    return value&&!std::strcmp(value,"1");\n'
            '}\n'
            'struct State{\n'
        ),
        (
            '    explicit State(const llama_model * m):model(m){}\n',
            '    // Fixed for the context lifetime. B holds the full bank and pools,\n'
            '    // but builds the original graph; it is not a runtime mode switch.\n'
            '    const bool trial_keep_original_graph;\n'
            '    explicit State(const llama_model * m):model(m),trial_keep_original_graph(original_graph_trial_flag()){}\n'
        ),
        (
            '    auto s=lookup(outer_cpu);if(!s)return nullptr;\n',
            '    auto s=lookup(outer_cpu);if(!s||s->trial_keep_original_graph)return nullptr;\n'
        ),
    ]
    for before, after in changes:
        if new.count(before) != 1:
            raise SystemExit("Expected unique source anchor is missing.")
        new = new.replace(before, after, 1)
    out.mkdir(parents=True, exist_ok=True)
    patch = ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                       fromfile='a/'+REL,tofile='b/'+REL))
    path = out/'optional_dormant_bank_original_graph.patch'
    path.write_text(patch)
    # Read-only applicability verification, not application or compilation.
    check = subprocess.run(['git','apply','--check',str(path)],cwd=repo,
                           capture_output=True,text=True)
    if check.returncode:
        raise SystemExit(check.stderr)
    status = subprocess.run(['git','status','--porcelain'],cwd=repo,
                            capture_output=True,text=True,check=True).stdout
    manifest = {
        'source_commit':PIN,'source':REL,'source_sha256':SOURCE_SHA,
        'proposed_source_sha256':hashlib.sha256(new.encode()).hexdigest(),
        'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),
        'status':'optional source-only diagnostic proposal, not deployed',
        'flag':'GGML_XEON_HOT_TRIAL_KEEP_ORIGINAL_GRAPH',
        'default':'unset or 0 preserves existing behavior',
        'one':'keep create/finish/native bank/private graph warmup, return original graph at build',
        'lifetime':'read once by State constructor; not runtime mutable',
        'unchanged':['plan validation','Gpre/Gpost and phase','bank/pinned buffers/private pools',
                     'C/S/P arithmetic','input/output ownership and synchronization'],
        'caller_work_still_required':['add explicit B-held arm','expect hot backend present',
                                     'record bank/pool identities','expect zero custom calls/builds',
                                     'do not pass B to C/S/P set_mode','F/B/C output comparison'],
        'git_apply_check':check.returncode==0,'repo_clean':not bool(status),
        'compiled':False,'engine_run':False,'numerically_verified':False,
        'performance_measured':False,
    }
    (out/'DORMANT_BANK_PATCH_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
