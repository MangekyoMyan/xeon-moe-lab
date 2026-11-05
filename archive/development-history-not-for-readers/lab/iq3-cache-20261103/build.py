#!/usr/bin/env python3
"""Isolated CPU/model overlays. Never edit production source, objects, or libraries."""
from pathlib import Path
import difflib, hashlib, json, shlex, subprocess, uuid
R=Path(__file__).resolve().parent;L=R.parent
H=L/'mxfp4-hybrid-20261102/hybrid';S=H/'engine/ggml/src';B=H/'build-cpu-ivybridge/engine/ggml/src'
E=L/'deepseek41-bringup-20261101';G=L/'gpu-stream-20261103';out=R/'runtime'
assert hashlib.sha256((H/'build-cpu-ivybridge/bin/libggml-cpu.so.0.28.0').read_bytes()).hexdigest()=='ff0b0f685bc469d82bf30de0644334bd3db94c596c0e633a35ef18d7890204ce'
commands=[]
def patch(path, changes):
    old=path.read_text();s=old
    for a,b in changes:
        if a.startswith(("    const bool src1_cont,", "    ggml_vec_dot_t    const vec_dot")):
            start=s.index("static void ggml_compute_forward_mul_mat_id_one_chunk(")
            end=s.index("\n#if defined(__AVX__)",start)
            chunk=s[start:end];assert chunk.count(a)==1
            s=s[:start]+chunk.replace(a,b)+s[end:]
        else:
            assert s.count(a)==1,(path,a[:120],s.count(a))
            s=s.replace(a,b)
    dest=R/path.name;dest.write_text(s)
    (R/(path.name+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(path),tofile=str(dest))))
    return dest
cpu=patch(S/'ggml-cpu/ggml-cpu.c',[
 ('#include "ggml-backend.h"','#include "ggml-backend.h"\n#include "iq3-cache.h"'),
 ('    const bool src1_cont,\n    const void * wdata) {\n\n    GGML_TENSOR_BINARY_OP_LOCALS', '    const bool src1_cont,\n    const void * wdata,\n    const void * compact_data) {\n\n    GGML_TENSOR_BINARY_OP_LOCALS'),
 ('    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;\n    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;\n\n    const int64_t blck_0 = 16;', '    ggml_vec_dot_t const vec_dot = compact_data ? xeon_iq3_compact_dot : type_traits_cpu[type].vec_dot;\n    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;\n    const size_t weight_stride = compact_data ? (size_t)(ne00/256)*144 : nb01;\n    const char * weight_data = compact_data ? (const char *) compact_data : src0_cur;\n\n    const int64_t blck_0 = 16;'),
 ('vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);','vec_dot(ne00, &tmp[ir0 - iir0], 0, weight_data + ir0*weight_stride, 0, src1_col, 0, 1);'),
 ('    for (int cur_a = 0; cur_a < n_as; ++cur_a) {\n        const int64_t cne1', '    const char * compact_data = (!params->use_ref && type == GGML_TYPE_IQ3_XXS && ids->ne[1] == 1)\n        ? (const char *) xeon_iq3_cache_lookup(src0) : NULL;\n\n    for (int cur_a = 0; cur_a < n_as; ++cur_a) {\n        const int64_t cne1'),
 ('if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1,', 'if (!compact_data && ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1,'),
 ('                    src0_cur, matrix_rows, row_size, src1_cont, wdata\n                );\n            }\n\n            if (nth >= nchunk0 * nchunk1)', '                    src0_cur, matrix_rows, row_size, src1_cont, wdata,\n                    compact_data ? compact_data + (size_t)cur_a*ne01*(ne00/256)*144 : NULL\n                );\n            }\n\n            if (nth >= nchunk0 * nchunk1)')])
backend=patch(S/'ggml-cpu/ggml-cpu.cpp',[
 ('#include "ggml-backend.h"','#include "ggml-backend.h"\n#include "iq3-cache.h"'),
 ('static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {','static void * ggml_backend_cpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {\n'+''.join(f'    if (strcmp(name, "{n}") == 0) return (void *) {n};\n' for n in ['xeon_iq3_cache_create_v1','xeon_iq3_cache_add_v1','xeon_iq3_cache_free_v1','xeon_iq3_cache_used_v1','xeon_iq3_cache_hits_v1','xeon_iq3_cache_total_v1']))])
model=patch(G/'llama-model.cpp',[
 ('#include "expert-host-register.h"','#include "expert-host-register.h"\n#include "model-cache.h"'),
 ('    ~impl() { expert_dma.clear(); }','    ~impl() { iq3_compact.clear(); expert_dma.clear(); }\n\n    xeon_iq3::cache iq3_compact;'),
 ('    load_tensors_hadamard();\n\n','    load_tensors_hadamard();\n\n'+(R/'model-cache-body.inc').read_text()+'\n')])
def flags(folder,name):
    result={}
    for line in (folder/f'CMakeFiles/{name}.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);result[k]=shlex.split(v)
    return result
cf=flags(B,'ggml-cpu')
def compile(src,f,extra=[]):
    lang='C' if src.suffix=='.c' else 'CXX'
    obj=R/(src.name+'.o')
    cmd=['/usr/bin/cc' if lang=='C' else '/usr/bin/c++',*f[lang+'_DEFINES'],*f[lang+'_INCLUDES'],*f[lang+'_FLAGS'],
         '-fno-fast-math','-ffp-contract=off','-I'+str(R),*extra,'-c',str(src),'-o',str(obj)]
    commands.append(cmd);subprocess.run(cmd,check=True,timeout=180);return obj
objs={}
for src in [cpu,backend]:objs['CMakeFiles/ggml-cpu.dir/ggml-cpu/'+src.name+'.o']=compile(src,cf)
new=compile(R/'iq3-cache.cpp',cf)
def link(folder,name,changes,extras,target):
    cmd=shlex.split((folder/f'CMakeFiles/{name}.dir/link.txt').read_text())
    for a,b in changes.items():
        assert cmd.count(a)==1,a
        cmd[cmd.index(a)]=str(b)
    stage=out/(target+'.'+uuid.uuid4().hex)
    cmd[cmd.index('-o')+1]=str(stage)
    cmd=[('-Wl,--dependency-file='+str(R/(name+'-link.d'))) if x.startswith('-Wl,--dependency-file=') else x for x in cmd]
    cmd.extend(map(str,extras));commands.append({'cwd':str(folder),'argv':cmd});subprocess.run(cmd,cwd=folder,check=True,timeout=90)
    stage.replace(out/target)
    for alias in [name+'.so',name+'.so.0']:
        alias=out/('lib'+alias)
        if not alias.exists():alias.symlink_to(target)
link(B,'ggml-cpu',objs,[new],'libggml-cpu.so.0.28.0')
mb=E/'build/src';mf=flags(mb,'llama')
mo=compile(model,mf,['-I'+str(G),'-I'+str(E/'source/src')])
link(mb,'llama',{'CMakeFiles/llama.dir/llama-model.cpp.o':mo},[],'libllama.so.0.8.0')
manifest={'commands':commands,'libraries':[{'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.so.*')) if not p.is_symlink()],
 'sources':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['iq3-cache.cpp','iq3-cache.h','compact-kernel.inc','ggml-cpu.c','ggml-cpu.cpp','llama-model.cpp','model-cache.h','model-cache-body.inc']},'production_modified':False}
(R/'BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n');print('CPU and model overlays built; production untouched',flush=True)
