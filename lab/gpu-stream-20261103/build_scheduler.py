#!/usr/bin/env python3
"""Build an opt-in MXFP4 scheduling threshold without changing CUDA kernels."""
from pathlib import Path
import difflib
import hashlib
import json
import shlex
import subprocess
import uuid

R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
SRC=ENGINE/'source/ggml/src';BUILD=ENGINE/'build/ggml/src'
original=(SRC/'ggml-backend.cpp').read_text()
helper='''static bool ggml_backend_sched_offload_batch_ok(const ggml_tensor * tensor) {
    static const int64_t min_mxfp4 = []() -> int64_t {
        const char * text = getenv("GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH");
        if (!text || !text[0]) return 0;
        char * end = nullptr;
        const long value = strtol(text, &end, 10);
        return end && !*end && value >= 1 && value <= 4096 ? value : 0;
    }();
    return tensor->op != GGML_OP_MUL_MAT_ID || tensor->src[0]->type != GGML_TYPE_MXFP4 ||
        tensor->src[1]->ne[2] >= min_mxfp4;
}

'''
anchor='static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {'
condition='if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {'
assert original.count(anchor)==1 and original.count(condition)==1
text=original.replace(anchor,helper+anchor).replace(condition,condition.replace(')) {',') && ggml_backend_sched_offload_batch_ok(tensor)) {'))
source=R/'ggml-backend.cpp';source.write_text(text)
(R/'scheduler.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/ggml/src/ggml-backend.cpp',tofile='b/ggml/src/ggml-backend.cpp')))
flags={}
for line in (BUILD/'CMakeFiles/ggml-base.dir/flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        key,value=line.split(' = ',1);flags[key]=shlex.split(value)
obj=R/'ggml-backend.cpp.o'
compile_cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-c',str(source),'-o',str(obj)]
link=shlex.split((BUILD/'CMakeFiles/ggml-base.dir/link.txt').read_text())
old='CMakeFiles/ggml-base.dir/ggml-backend.cpp.o';assert link.count(old)==1;link[link.index(old)]=str(obj)
target=R/'runtime/libggml-base.so.0.28.0';stage=target.with_name('base-stage-'+uuid.uuid4().hex+'.so')
link[link.index('-o')+1]=str(stage)
link=[('-Wl,--dependency-file='+str(R/'base-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
subprocess.run(compile_cmd,check=True,timeout=120);subprocess.run(link,cwd=BUILD,check=True,timeout=60)
stage.replace(target)
for name in ['libggml-base.so.0','libggml-base.so']:
    alias=target.with_name(name)
    if not alias.exists():alias.symlink_to(target.name)
(R/'SCHEDULER.json').write_text(json.dumps({'library':str(target),'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'compile':compile_cmd,'link':link,'production_modified':False},indent=2)+'\n')
print('Built isolated scheduler threshold overlay')
