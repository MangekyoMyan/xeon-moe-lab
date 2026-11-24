"""Rebuild just the CPU translation unit using the active library's original objects and flags."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from shared import HERE, LAB, PRIOR, sha, fresh, read

BASE = LAB/"exploration-20261112/mx-prepack"
SOURCE = BASE/"backend/ggml-cpu.c"
RECIPE = BASE/"release-candidate-a/BUILD.json"
ORIGINAL = BASE/"release-candidate-a/libggml-cpu.so.0.28.0"
CPU = HERE/"cpu"


def once(src, before, after):
    if src.count(before) != 1:
        raise ValueError(f"expected one source anchor: {before[:100]!r} (found {src.count(before)})")
    return src.replace(before, after)


def traced(src):
    src=once(src,"// ggml_compute_forward_mul_mat_id\n",'#include "worker_trace.inc"\n\n// ggml_compute_forward_mul_mat_id\n')
    src=once(src,"    ggml_barrier(params->threadpool);\n\n    uint64_t mx_safe_mask",
        "    if (wb_current) wb_current->prep_in=wb_clock();\n    ggml_barrier(params->threadpool);\n    if (wb_current) wb_current->prep_out=wb_clock();\n\n    uint64_t mx_safe_mask")
    src=once(src,"        // tiled takes over if profitable for this expert",'''        struct wb_expert *observed = NULL;
        if (wb_current) {
            GGML_ASSERT(wb_current->nexperts < WB_EXPERTS);
            observed=&wb_current->e[wb_current->nexperts++];
            observed->id=cur_a; observed->compact=compact_data!=NULL; observed->inputs=cne1;
            observed->start=wb_clock();
        }

        // tiled takes over if profitable for this expert''')
    src=once(src,"tiled_scratch)) {\n            continue;\n        }\n\n        const char * src0_cur",
                "tiled_scratch)) {\n            if (observed) { observed->tiled=1; observed->end=wb_clock(); }\n            continue;\n        }\n\n        const char * src0_cur")
    anchor="            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);\n\n#if defined(__AVX__)"
    src=once(src,anchor,anchor.replace("\n\n#if","\n            if (observed) { observed->chunks++; observed->rows+=(ir0_end-ir0_start)*(ir1_end-ir1_start); }\n\n#if"))
    anchor="            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);\n        }\n    }\n}"
    src=once(src,anchor,anchor.replace("        }\n    }","        }\n        if (observed) observed->end=wb_clock();\n    }"))
    src=once(src,"    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort",
        "    const int pilot_graph=atomic_load_explicit(&wb_on,memory_order_acquire) ? atomic_load_explicit(&wb_graph,memory_order_acquire) : 0;\n    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort")
    anchor="        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);"
    src=once(src,anchor,"        struct wb_record *wr=wb_node(pilot_graph,node_n,state->ith,params.nth,node);\n"+anchor)
    anchor="""        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }"""
    src=once(src,anchor,"""        const bool has_barrier = node_n + 1 < cgraph->n_nodes;
        if (wr) { wr->end_cpu=sched_getcpu(); wr->sync=has_barrier; wr->finish=wb_clock(); }
        if (has_barrier) ggml_barrier(state->threadpool);
        if (wr) { wr->released=wb_clock(); wb_current=NULL; }
    }""")
    anchor="    ggml_barrier(state->threadpool);\n\n#ifdef GGML_USE_CPU_RISCV64_SPACEMIT"
    src=once(src,anchor,"""    struct wb_record *wr_end=wb_node(pilot_graph,cgraph->n_nodes,state->ith,params.nth,NULL);
    if (wr_end) { wr_end->end_cpu=sched_getcpu(); wr_end->sync=1; wr_end->finish=wb_clock(); }
    ggml_barrier(state->threadpool);
    if (wr_end) { wr_end->released=wb_clock(); wb_current=NULL; }

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT""")
    anchor="    bool disposable_threadpool = false;"
    src=once(src,anchor,"""    const bool observed_graph=atomic_load_explicit(&wb_on,memory_order_acquire)!=0;
    if (observed_graph) {
        GGML_ASSERT(atomic_exchange(&wb_graph_busy,1)==0);
        atomic_fetch_add(&wb_graph,1);
    }
"""+anchor)
    anchor="    enum ggml_status ret = threadpool->ec;"
    return once(src,anchor,"    if (observed_graph) atomic_store(&wb_graph_busy,0);\n"+anchor)


def cpu_build(kind):
    assert sha(SOURCE)=="e8fdf3d424463f9852fa774d9de6057dce35855a872936bda87d12d3d756e3a8"
    assert sha(ORIGINAL)=="33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3"
    base=SOURCE.read_text()
    if kind=="rebuild": modified=base
    elif kind=="trace":modified=traced(base)
    else:
        from proposals import candidate
        modified=candidate(base,kind)
    d=CPU/kind;d.mkdir()
    source=d/"ggml-cpu.c";source.write_text(modified)
    (d/"change.patch").write_text("".join(difflib.unified_diff(base.splitlines(True),modified.splitlines(True),fromfile="a/ggml-cpu.c",tofile="b/ggml-cpu.c")))
    receipt=read(RECIPE)["variants"]["candidate"]
    cc=receipt["compile"].copy();oldobj=cc[cc.index("-o")+1]
    cc[cc.index("-o")+1]=str(d/"ggml-cpu.c.o"); cc[cc.index("-c")+1]=str(source)
    cc[1:1]=["-I"+str(CPU),"-ffile-prefix-map="+str(source)+"="+str(SOURCE)]
    runtime=d/"runtime";runtime.mkdir()
    lib=runtime/ORIGINAL.name
    link=receipt["link"]["argv"].copy()
    assert link.count(oldobj)==1
    link[link.index(oldobj)]=str(d/"ggml-cpu.c.o")
    link[link.index("-o")+1]=str(lib)
    for i,x in enumerate(link):
        if x.startswith("-Wl,--dependency-file="):link[i]="-Wl,--dependency-file="+str(d/"link.d")
    with (d/"build.log").open("w") as log:
        subprocess.run(cc,stdout=log,stderr=log,check=True,timeout=180)
        subprocess.run(link,cwd=receipt["link"]["cwd"],stdout=log,stderr=log,check=True,timeout=120)
    if kind=="rebuild":
        assert sha(d/"ggml-cpu.c.o")==sha(Path(oldobj)) and sha(lib)==sha(ORIGINAL)
    for name in ("libggml-cpu.so","libggml-cpu.so.0"):(runtime/name).symlink_to(lib.name)
    keep={}
    for token in link:
        p=Path(token)
        if not p.is_absolute():p=Path(receipt["link"]["cwd"])/p
        if p.suffix in (".o",".a") and p.is_file():keep[str(p)]=sha(p)
    fresh(d/"BUILD.json",{"kind":kind,"original_library":sha(ORIGINAL),"source_sha256":sha(source),
          "included_trace_sha256":sha(CPU/"worker_trace.inc") if kind=="trace" else None,
          "patch_sha256":sha(d/"change.patch"),"compile":cc,"link":link,"link_cwd":receipt["link"]["cwd"],
          "input_objects":keep,"library":str(lib),"sha256":sha(lib)})
    print(kind,sha(lib),flush=True)


def driver_build():
    old=PRIOR/"driver";d=HERE/"driver";d.mkdir()
    for n in ("CASES.json","affinity_check.h","cache_inspection.h","tracing.h"):shutil.copy2(old/n,d/n)
    s=(old/"driver.cpp").read_text()
    anchor="        const bool traced_stage = stage==\"observe\";"
    s=once(s,anchor,anchor+'''
        const bool worker_trace=spec.value("worker_trace",false);
        auto wb_begin=reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT,"xeon_worker_trace_begin"));
        auto wb_end=reinterpret_cast<int(*)(const char*)>(dlsym(RTLD_DEFAULT,"xeon_worker_trace_end"));
        if (worker_trace) {
            Dl_info wi{},wj{};
            need(wb_begin && wb_end && dladdr(reinterpret_cast<void*>(wb_begin),&wi) &&
                 dladdr(reinterpret_cast<void*>(wb_end),&wj) && fs::canonical(wi.dli_fname)==fs::canonical(wj.dli_fname)
                 && fs::canonical(wi.dli_fname)==fs::path(spec.at("cpu_path").get<std::string>()), "worker trace DSO");
        }''')
    anchor="                starts[i]=ns();"
    s=once(s,anchor,'''                const bool sample=worker_trace && (i%8==4);
                if (sample) wb_begin();
'''+anchor)
    anchor="                ends[i]=ns();"
    s=once(s,anchor,anchor+'''
                if (sample) {
                    auto dest=out/("worker-"+std::to_string(c)+"-"+std::to_string(i+1)+".tsv");
                    const int count=wb_end(dest.c_str());
                    need(count>0, "CPU trace write");
                    report["worker_graphs"][std::to_string(c)][std::to_string(i+1)]=count;
                }''')
    (d/"driver.cpp").write_text(s)
    cmd=read(old/"BUILD.json")["command"]
    for i,x in enumerate(cmd):
        if x in (str(old/"driver"),str(old/"driver.cpp"),"-I"+str(old)):cmd[i]=x.replace(str(old),str(d))
    with (d/"build.log").open("w") as log:subprocess.run(cmd,stdout=log,stderr=log,timeout=240,check=True)
    fresh(d/"BUILD.json",{"command":cmd,"driver_sha256":sha(d/"driver"),"source_sha256":{n:sha(d/n) for n in ("driver.cpp","CASES.json","affinity_check.h","cache_inspection.h","tracing.h")}})
    print("driver",sha(d/"driver"),flush=True)


if __name__=="__main__":
    if sys.argv[1]=="driver":driver_build()
    else:cpu_build(sys.argv[1])
