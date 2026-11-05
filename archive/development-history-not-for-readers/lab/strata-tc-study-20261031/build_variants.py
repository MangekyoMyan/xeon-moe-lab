#!/usr/bin/env python3
import hashlib,json,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;LAB=R.parent
BASE=LAB/'prefill-study-20261030/release/prefill-v1'
STREAM=LAB/'moe-engine-20261029/experiments/stream-active-agent/v2/variant'
CUDA=LAB/'v100-prep-20261024/cuda-12.9.2';E=LAB/'xeon-volta-engine'
decl='''extern "C" int xv_q8_tc_cuda(const void*,const void*,const int32_t*,const int32_t*,float*,int,int,int,int,int,void*);\n'''
for name,parent in [('hybrid-tc',BASE),('stream-tc',STREAM)]:
    v=R/'variants'/name;s=v/'source';rt=v/'runtime';s.mkdir(parents=True,exist_ok=True);rt.mkdir(exist_ok=True)
    for p in parent.joinpath('source').iterdir():
        if p.is_file():
            dst=s/p.name
            if dst.exists():dst.chmod(0o644)
            shutil.copy2(p,dst);dst.chmod(0o644)
    for p in BASE.joinpath('runtime').iterdir():
        if p.name in ['libxv-experiment.so','libxv-q8-exact.so']:continue
        if not (rt/p.name).exists():(rt/p.name).symlink_to(p.resolve())
    for n in ['tc-dispatch.cu','tc-q8.cuh']:shutil.copy2(R/'source'/n,s/n)
    # Keep the accepted exact fallback, including decoder arithmetic, identical.
    shutil.copy2(R/'source/exact-baseline.cu',s/'q8-exact.cu')
    p=s/'expert-cache.cpp';text=p.read_text();needle='#include "q8-exact.h"\n';assert text.count(needle)==1;text=text.replace(needle,needle+decl)
    old='''GGML_ASSERT(xv_q8_exact_cuda(weights->data,graph->input->data,(int32_t *)graph->ids->data,(float *)output->data,
                            int(weights->ne[0]),int(weights->ne[1]),int(weights->ne[2]),int(ggml_nelements(graph->ids)),
                            int(ggml_nelements(graph->input)/graph->input->ne[0]),stream)==0);'''
    new='''if (xv_env_one("GGML_XV_TC_PREFILL") && phase_index == 1) {
                            GGML_ASSERT(xv_q8_tc_cuda(weights->data,graph->input->data,(int32_t *)graph->ids->data,graph->ids_data(),(float *)output->data,
                                int(weights->ne[0]),int(weights->ne[1]),int(weights->ne[2]),int(ggml_nelements(graph->ids)),
                                int(ggml_nelements(graph->input)/graph->input->ne[0]),stream)==0);
                        } else { '''+old+''' }'''
    assert text.count(old)==1;text=text.replace(old,new)
    # Streamed positions are already grouped by expert. A larger cap avoids
    # splitting a group's work into many tiny launches; same cap for exact/TC A/B.
    if name=='stream-tc':
        text=text.replace('position_first+=1024','position_first+=8192').replace('std::min<size_t>(1024,chunk_positions.size()-position_first)','std::min<size_t>(8192,chunk_positions.size()-position_first)')
    p.write_text(text)
    cmds=[]
    cmd=[str(CUDA/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_70','--fmad=false','--cudart','shared','-ccbin','/usr/bin/g++-14','-Xcompiler=-fPIC','-shared','-DGGML_CUDA_USE_GRAPHS','-I'+str(E/'source/ggml/include'),'-I'+str(E/'source/ggml/src'),str(s/'q8-exact.cu'),str(s/'q8-backend.cu'),str(s/'tc-dispatch.cu'),'-L'+str(rt),'-lggml-cuda','-lggml-base','-lcublas','-o',str(rt/'libxv-q8-exact.so')]
    cmds.append(cmd);subprocess.run(cmd,check=True)
    cmd=['g++','-std=c++17','-O2','-march=ivybridge','-fPIC','-shared','-Wall','-Wextra','-Werror','-pthread','-DXV_EXACT_Q8','-I'+str(E/'source/ggml/include'),'-I'+str(s),str(s/'expert-cache.cpp'),'-L'+str(rt),'-lggml-cuda','-lggml-base','-lxv-q8-exact','-lggml-cpu','-Wl,-rpath,$ORIGIN','-o',str(rt/'libxv-experiment.so')]
    cmds.append(cmd);subprocess.run(cmd,check=True)
    tracked=[*s.iterdir(),rt/'libxv-q8-exact.so',rt/'libxv-experiment.so']
    (v/'BUILD.json').write_text(json.dumps({'parent':str(parent),'commands':cmds,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked if p.is_file()}},indent=2)+'\n')
    print('Built',name,flush=True)
