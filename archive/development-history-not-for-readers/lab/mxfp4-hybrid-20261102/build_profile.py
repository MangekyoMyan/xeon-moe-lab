#!/usr/bin/env python3
"""Build only a diagnostic stock CPU library using the saved production flags/objects."""
from pathlib import Path
import difflib
import hashlib
import json
import shlex
import shutil
import subprocess

R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
out=R/'profile-stock';out.mkdir(exist_ok=False)
source=ENGINE/'source/ggml/src/ggml-cpu/ggml-cpu.c';original=source.read_text();modified=original
needle='static void ggml_compute_forward_mul_mat_id(\n';assert modified.count(needle)==1
modified=modified.replace(needle,'#if defined(GGML_MOE_HISTOGRAM) && defined(__linux__)\n#include "moe-histogram.h"\n#endif\n\n'+needle)
needle='    ggml_barrier(params->threadpool);\n\n    for (int cur_a = 0; cur_a < n_as; ++cur_a) {';assert modified.count(needle)==1
modified=modified.replace(needle,'    ggml_barrier(params->threadpool);\n\n#if defined(GGML_MOE_HISTOGRAM) && defined(__linux__)\n    if (ith == 0) moe_hist_record(params, dst, matrix_row_counts, n_as);\n#endif\n\n    for (int cur_a = 0; cur_a < n_as; ++cur_a) {')
(out/'ggml-cpu.c').write_text(modified);shutil.copy2(R/'kernels/moe-histogram.h',out/'moe-histogram.h')
patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/ggml/src/ggml-cpu/ggml-cpu.c',tofile='b/ggml/src/ggml-cpu/ggml-cpu.c'))
(R/'profile-stock.patch').write_text(patch)
target=ENGINE/'build/ggml/src/CMakeFiles/ggml-cpu.dir';cwd=target.parent.parent
flags={}
for line in (target/'flags.make').read_text().splitlines():
    if ' = ' in line:
        name,value=line.split(' = ',1);flags[name]=shlex.split(value)
obj=out/'ggml-cpu-profile.o';lib=out/'libggml-cpu.so.0.28.0'
compile_cmd=['/usr/bin/cc',*flags['C_DEFINES'],*flags['C_INCLUDES'],*flags['C_FLAGS'],'-DGGML_MOE_HISTOGRAM',
             '-c',str(out/'ggml-cpu.c'),'-o',str(obj)]
link=shlex.split((target/'link.txt').read_text());link[link.index('-o')+1]=str(lib)
objects=[];replaced=0
for i,arg in enumerate(link):
    if arg.endswith('/ggml-cpu.c.o'):
        link[i]=str(obj);replaced+=1
    elif arg.startswith('-Wl,--dependency-file='):
        link[i]='-Wl,--dependency-file='+str(out/'link.d')
    elif arg.endswith('.o'):
        p=(cwd/arg).resolve();objects.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert replaced==1
subprocess.run(compile_cmd,cwd=cwd,check=True,timeout=60)
subprocess.run(link,cwd=cwd,check=True,timeout=30)
for name in ['libggml-cpu.so','libggml-cpu.so.0']:(out/name).symlink_to(lib.name)
record={'profile_type':'stock math plus optional aggregate histogram; no hybrid kernel',
 'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'profile_source_sha256':hashlib.sha256((out/'ggml-cpu.c').read_bytes()).hexdigest(),
 'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'compile':compile_cmd,'link':link,
 'link_cwd':str(cwd),'reused_objects':objects,'production_modified':False}
(out/'BUILD.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'profile_library':str(lib),'sha256':record['library_sha256']}),flush=True)
