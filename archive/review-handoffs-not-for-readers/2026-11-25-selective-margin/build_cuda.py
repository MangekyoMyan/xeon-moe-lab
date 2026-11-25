"""Build an isolated CUDA DSO with two changed translation units from the deployed recipe."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import time
from prepare import ROOT,LAB,load,save,sha

D=ROOT/"backend"; base=LAB/"exploration-20261112/cuda-sort"; engine=LAB/"deepseek41-bringup-20261101"
def replace(src,a,b):
    assert src.count(a)==1,(a[:100],src.count(a))
    return src.replace(a,b)


def write_source(path,target,fn):
    original=path.read_text();modified=fn(original)
    (D/target).write_text(modified)
    (D/(target+".patch")).write_text("".join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile="a/"+target,tofile="b/"+target)))


def cuda(src):
    src=replace(src,"static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {",
                    '#include "margin_core.inc"\n\nstatic void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {')
    key="    std::vector<char> ids_host(ggml_nbytes(ids));"
    src=replace(src,key,'''    const int margin_p=margin_projection(dst);
    margin_plan local_plan;
    if(margin_p>=0 && !marginal.host_ready){
        CUDA_CHECK(cudaMemcpyAsync(marginal.host,marginal.device,size_t(marginal.n)*sizeof(margin_row),cudaMemcpyDeviceToHost,stream));
    }
'''+key)
    key="    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());"
    src=replace(src,key,'''    if(margin_p>=0){
        marginal.host_ready=true;
        margin_build_plan(local_plan,margin_p,ids,ids_host.data(),ids_to_sorted_host,tokens_per_expert,stream);
    }
'''+key)
    key="        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);"
    src=replace(src,key,key+'''
        if(margin_p>=0){
            const int start=local_plan.starts[i02],end=local_plan.starts[i02+1];
            margin_correct(ctx,margin_p,src0_slice,src1_slice,dst_slice,start,end-start,cc);
        }''')
    key="        case GGML_OP_ARGSORT:\n            ggml_cuda_op_argsort(ctx, dst);"
    return replace(src,key,key+"\n            margin_after_argsort(ctx, dst);")


def topk(src):
    src=replace(src,'#include "topk-moe.cuh"','#include "topk-moe.cuh"\n#include "margin_shared.h"')
    src=replace(src,"struct topk_moe_config {","struct topk_moe_config {\n    margin_row *margin;\n    float cut;")
    src=replace(src,"template <int n_experts, bool has_bias>\n__launch_bounds__","template <int n_experts, bool has_bias, bool with_margin = false>\n__launch_bounds__")
    src=replace(src,"    float wt_sum = 0.f;","    float last_score=0.f; int last_id=-1;\n    float wt_sum = 0.f;")
    key="            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {\n                selection_wt[max_expert / WARP_SIZE] = -INFINITY;"
    src=replace(src,key,"            if constexpr(with_margin){last_score=max_val_s;last_id=max_expert;}\n"+key)
    key="    if (config.with_norm) {\n        wt_sum              = warp_reduce_sum(wt_sum);"
    src=replace(src,key,'#include "seventh.inc"\n'+key)
    key="        case 384: // DeepSeek V4.1\n"
    src=replace(src,key,key+'''            if constexpr(has_bias){
                if(config.margin){
                    ggml_cuda_kernel_launch(topk_moe_cuda<384, true, true>, launch_params,
                        logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
                    break;
                }
            }
''')
    key="    config.use_sigmoid       = args.sigmoid;"
    src=replace(src,key,'''    config.cut=0;config.margin=nullptr;
    if(bias && args.sqrt_softplus && args.n_groups==0 && n_experts==384 && n_expert_used==6)
        config.margin=xeon_margin_rows_for(ids->name,"ffn_moe_topk-",n_rows,&config.cut);
'''+key)
    return src


def main():
    recipe=load(base/"BUILD.json");installed=base/"release-candidate-a/libggml-cuda.so.0.28.0"
    assert sha(installed)==recipe["variants"]["candidate"]["sha256"]=="45f6b7d8334c415818f79b13add7c9710d552bbab2ff6c1f4c8bb5770f8de8f0"
    cp=base/"ggml-cuda.cu";tp=engine/"source/ggml/src/ggml-cuda/topk-moe.cu"
    assert sha(cp)==recipe["candidate_source_sha256"]
    write_source(cp,"ggml-cuda.cu",cuda);write_source(tp,"topk-moe.cu",topk)
    cwd=Path(recipe["cwd"]);runtime=D/"runtime";runtime.mkdir(exist_ok=True)
    lib=runtime/installed.name
    commands=[]
    for src in ("ggml-cuda.cu","topk-moe.cu"):
        cc=recipe["compile"].copy()
        cc.insert(1,"-I"+str(D))
        cc[cc.index("-c")+1]=str(D/src);cc[cc.index("-o")+1]=str(D/(src+".o"));commands.append(cc)
    link=recipe["variants"]["candidate"]["link"].copy()
    for a in list(link):
        if a.startswith("-Wl,--dependency-file="):link[link.index(a)]="-Wl,--dependency-file="+str(D/"link.d")
        elif a.endswith("ggml-cuda.cu.o"):
            assert sha(Path(a))==sha(base/"ggml-cuda.cu.o");link[link.index(a)]=str(D/"ggml-cuda.cu.o")
        elif a.endswith("topk-moe.cu.o"):
            p=(cwd/a).resolve();assert sha(p)==recipe["original_link_inputs"][str(p)]
            link[link.index(a)]=str(D/"topk-moe.cu.o")
    link[link.index("-o")+1]=str(lib);commands.append(link)
    start=time.monotonic()
    with (D/"build.log").open("w") as log:
        for cmd in commands:subprocess.run(cmd,cwd=cwd,stdout=log,stderr=log,check=True,timeout=600)
    for n in ("libggml-cuda.so","libggml-cuda.so.0"):
        p=runtime/n
        if not p.exists():p.symlink_to(lib.name)
    names=subprocess.check_output(["nm","-D",str(lib)],text=True)
    assert all(" T xeon_margin_"+v in names for v in ("open","begin","receipt","close"))
    save(D/"BUILD.json",{"sources":{n:sha(D/n) for n in ("ggml-cuda.cu","topk-moe.cu","margin_core.inc","seventh.inc","margin_shared.h")},
                         "parents":{str(p):sha(p) for p in (cp,tp,base/"BUILD.json")},"commands":commands,"cwd":str(cwd),
                         "library":str(lib),"sha256":sha(lib),"ordinary":sha(installed),"seconds":time.monotonic()-start})
    print("isolated CUDA backend",sha(lib),flush=True)


if __name__=="__main__":main()
