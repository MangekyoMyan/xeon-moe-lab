#!/usr/bin/env python3
"""Build isolated libllama overlay using original flags and unchanged CUDA objects."""
from pathlib import Path
import difflib
import hashlib
import json
import shlex
import subprocess
import uuid

R=Path(__file__).resolve().parent;ENGINE=R.parent/'deepseek41-bringup-20261101'
SRC=ENGINE/'source/src';BUILD=ENGINE/'build/src'
original=(SRC/'llama-model.cpp').read_text();text=original
def replace(old,new):
    global text
    assert text.count(old)==1,old
    text=text.replace(old,new)
replace('#include "llama-model.h"', '#include "llama-model.h"\n#include "expert-host-register.h"\n#include <cstdlib>\n#if defined(__linux__)\n#include <unistd.h>\n#endif')
replace('    ~impl() = default;', '    ~impl() { expert_dma.clear(); }\n\n    xeon_expert_dma::registrations expert_dma;')
replace('    load_tensors_hadamard();\n\n    return true;', '    load_tensors_hadamard();\n\n'+(R/'registration-body.inc').read_text()+'\n    return true;')
source=R/'llama-model.cpp';source.write_text(text)
(R/'libllama.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/src/llama-model.cpp',tofile='b/src/llama-model.cpp')))
flags={}
for line in (BUILD/'CMakeFiles/llama.dir/flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        key,value=line.split(' = ',1);flags[key]=shlex.split(value)
obj=R/'llama-model.cpp.o'
compile_cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(SRC),'-c',str(source),'-o',str(obj)]
out=R/'runtime';out.mkdir(exist_ok=True)
link=shlex.split((BUILD/'CMakeFiles/llama.dir/link.txt').read_text())
assert link.count('CMakeFiles/llama.dir/llama-model.cpp.o')==1
link[link.index('CMakeFiles/llama.dir/llama-model.cpp.o')]=str(obj)
target=out/'libllama.so.0.8.0'
staged=out/('libllama-stage-'+uuid.uuid4().hex+'.so')
link[link.index('-o')+1]=str(staged)
link=[('-Wl,--dependency-file='+str(R/'libllama-link.d')) if x.startswith('-Wl,--dependency-file=') else x for x in link]
subprocess.run(compile_cmd,check=True,timeout=180)
subprocess.run(link,cwd=BUILD,check=True,timeout=90)
staged.replace(target)
for name in ['libllama.so.0','libllama.so']:
    alias=out/name
    if not alias.exists():alias.symlink_to(target.name)
manifest={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'helper_sha256':hashlib.sha256((R/'expert-host-register.h').read_bytes()).hexdigest(),
          'engine_revision':subprocess.check_output(['git','-C',str(ENGINE/'source'),'rev-parse','HEAD'],text=True).strip(),
          'original_library_sha256':hashlib.sha256((ENGINE/'build/bin/libllama.so.0.8.0').read_bytes()).hexdigest(),
          'original_source_sha256':hashlib.sha256((SRC/'llama-model.cpp').read_bytes()).hexdigest(),
          'library':str(target),'library_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
          'compile':compile_cmd,'link':link,'link_cwd':str(BUILD),'production_modified':False}
(R/'OVERLAY.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'built':str(target),'production_modified':False}),flush=True)
