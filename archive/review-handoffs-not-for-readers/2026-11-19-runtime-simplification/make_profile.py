"""Describe one verified x9ram runtime using the local hardware and model receipts."""
import argparse
import json
from pathlib import Path
from live import LAB, ENGINE, digest, read

ROOT=Path(__file__).resolve().parent

def asset(path):
    p=Path(path).resolve()
    return {"file":str(p.relative_to(LAB)), "sha256":digest(p)}

def describe(live, output):
    cfg_file=ENGINE/"server-config.json"
    cfg=read(cfg_file)
    assert digest(cfg_file)==live["config_sha256"]
    args=cfg["command"]
    env=cfg["environment"]
    def arg(name):return args[args.index(name)+1]
    def relative(p):return str(Path(p).relative_to(LAB))
    recipe_steps={
        "cpu":"build/BUILD.json; reused object compile commands in CPU_OBJECT_PROVENANCE.json",
        "llama":"exploration-20261112/sparse-cache/LOADER_BUILD.json",
        "cuda":"exploration-20261112/cuda-sort/release-candidate-a/BUILD.json",
        "base":"gpu-stream-20261103/SCHEDULER.json",
    }
    libraries={}
    for key,soname in [("cpu","libggml-cpu.so.0.28.0"),("llama","libllama.so.0.8.0"),("cuda","libggml-cuda.so.0.28.0"),("base","libggml-base.so.0.28.0")]:
        path=next(Path(p) for p in live["mapped_runtime_hashes"] if Path(p).name==soname)
        libraries[key]=asset(path)
        assert libraries[key]["sha256"]==live["mapped_runtime_hashes"][str(path)]
        libraries[key]["build_record"]=recipe_steps[key]
    model_dir=Path(cfg["model"]).parent.parent
    mr=model_dir/"download-verified.json"
    model=read(mr)
    vr=Path(cfg["vision_profile"]["projector"]).parent/"download-verified.json"
    vv=read(vr)
    model_files=[{"file":relative(model_dir/f["name"]), "size":f["size"],"sha256":f["sha256"]} for f in model["files"]]
    vision_files=[{"file":relative(vr.parent/vv["file"]),"size":vv["size"],"sha256":vv["sha256"]}]
    assert all((LAB/f["file"]).stat().st_size==f["size"] for f in model_files+vision_files)
    topo=read(ROOT/"INITIAL_TOPOLOGY.json")
    memory=read(ROOT/"DIMMS.json")
    plan=asset(env["GGML_CPU_IQ_PLAN"])
    header=(LAB/plan["file"]).open().readline().split()
    plan.update(format=header[0],tensor_count=int(header[1]),selected_projections=int(header[2]),packed_bytes=int(header[3]),
                ram_budget_bytes=int(env["GGML_CPU_IQ3_COMPACT_MIB"])*1024**2,model_download_fingerprint=digest(mr),
                published_copy="models/deepseek41-mix-256g/frequency19200.plan")
    chosen=read(ROOT/"build/BUILD.json")["ordinary"]["sha256"]==libraries["cpu"]["sha256"]
    assert chosen or libraries["cpu"]["sha256"]=="33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3"
    cfgp=asset(cfg_file)
    info={
        "schema":1,"profile":"E5-2630L v2 x2 · 512 GiB · PG500-216 · DeepSeek Mix-256G","verified_jst_date":"2026-11-19",
        "hardware":{"cpu":live["CPU_model"],"sockets":len({v["socket"] for v in topo["cores"]}),"physical":topo["physical_cores_online"],"logical":topo["logical_cpus_online"],
                    "one_cpu_per_physical_core":topo["one_available_thread_per_core"],"cpu_topology":topo["cores"],
                    "board":(Path("/sys/class/dmi/id/board_vendor").read_text().strip()+" "+Path("/sys/class/dmi/id/board_name").read_text().strip()),"bios":Path("/sys/class/dmi/id/bios_version").read_text().strip(),
                    "populated_dimms":len(memory),"dimm_sizes":sorted({r["Size"] for r in memory}),"configured_memory_speeds":sorted({r["Configured Memory Speed"] for r in memory}),"ram_GiB_installed":512,
                    "GPU":"PG500-216 (Volta, sm70, 32 GB class)","GPU_limit_W":live["gpu_watts_limit"],"gpu_ecc":live["gpu_ecc"]},
        "inference":{"context":int(arg("-c")),"threads_decode":int(arg("-t")),"threads_prefill":int(arg("-tb")),"decode_cpu_mask":arg("--cpu-mask"),"prefill_cpu_mask":arg("--cpu-mask-batch"),
                     "OMP_THREAD_LIMIT":int(env["OMP_THREAD_LIMIT"]),"batch":int(arg("-b")),"ubatch":int(arg("-ub")),"kv_type":[arg("--cache-type-k"),arg("--cache-type-v")],
                     "GPU_routed_layers":cfg["gpu_resident_expert_layers"],"cpu_routed_count":int(arg("--n-cpu-moe")),"host_weight_registration":env["GGML_CUDA_REGISTER_HOST"],
                     "interleave_nodes":[0,1],"THP_disabled_by_wrapper":True,"offload_min_batch":int(env["GGML_OP_OFFLOAD_MIN_BATCH"]),"mxfp4_offload_min_batch":int(env["GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH"]),
                     "packing_threads_at_load":12,"compact_packing_host_reserve_bytes":64*1024**3,
                     "load_mode":arg("--load-mode"),"lazy_mode":arg("--lazy-mode"),"fan_mode":"Standard","fan_limit_percent":70,"image_max_tokens":int(arg("--image-max-tokens")),"parallel_slots":int(arg("--parallel"))},
        "service_config":cfgp,"server_executable":asset(ENGINE/"build/bin/llama-server"),"service":{"unit":"llm-deepseek41-api.service","start_on_x9ram":"sudo python3 "+str(ENGINE/"start.py"),"status_on_x9ram":"sudo python3 "+str(ENGINE/"status.py")},
        "libraries":libraries,"plan":plan,"upstream_revision":cfg["source_revision"],"ld_library_path":env["LD_LIBRARY_PATH"].split(":"),
        "external_libraries_from_proc_maps":[p for p in live["dependencies_mapped"] if Path(p).name not in {Path(v["file"]).name for v in libraries.values()}],
        "model":{"repository":model["repository"],"revision":model["revision"],"quantization":"Mix-256G (mixed tensor quantization)","first_shard":relative(cfg["model"]),
                 "files":model_files,"download_receipt":asset(mr)},"vision":{"repository":vv["repository"],"revision":vv["revision"],"files":vision_files,"download_receipt":asset(vr)},
        "features":{"accepted":["IQ compact with native fallback","MXFP4 Q8 input preparation reuse","stable CUDA expert ID sorting","host expert weight registration"],
                    "model_specific":["quantization and tokenizer revision","frequency19200.plan and image projector"],"hardware_specific":["worker counts and affinity","NUMA and RAM/VRAM budgets","offload thresholds and expert placement"],
                    "closed_experiments":["shared GPU prefix (ecc9)","CPU08","additional hot expert bank"],"historical":"2026-11-05 accepted/ checkpoint; compact budget 96 GiB"},
        "row_local":{"normal_build_contains_specialized_placement":not chosen,"normal_enabled":False,"experiment_patch":"experiments/row-local/enable.patch","experiment_base":"cpu/iq3-cache.cpp"},
        "rollback":{"baseline_config":asset(LAB/"exploration-20261112/profiles/final.json"),"baseline_cpu":asset(LAB/"exploration-20261112/mx-prepack/release-candidate-a/libggml-cpu.so.0.28.0")},
        "planned_cpu_upgrade":{"model":"E5-2697 v2 x2","status":"planned, no measurements","expected_physical":24,"expected_logical":48,"next":"Run check.py topology on the new hardware; measure worker counts, then offload thresholds and compact allocation independently."}
    }
    for source in [LAB/"exploration-20261112/row-local/iq3-cache.cpp", LAB/"exploration-20261112/row-local/placement.inc", LAB/"exploration-20261112/mx-prepack/backend/ggml-cpu.c"]:
        libraries["cpu"].setdefault("original_source_files", []).append(asset(source))
    libraries["cpu"]["as_built_recipe"] = asset(LAB/"exploration-20261112/mx-prepack/release-candidate-a/BUILD.json")
    info["row_local"]["comparison_source"] = "cpu/iq3-cache.cpp"
    info["row_local"]["comparison_CPU_sha256"] = read(ROOT/"build/BUILD.json")["ordinary"]["sha256"]
    info["row_local"]["production_cache_source"] = "exploration-20261112/row-local/iq3-cache.cpp" if not chosen else "cpu/iq3-cache.cpp"
    if chosen:
        info["rollback"]["command"]="sudo python3 "+str(ROOT/"switch_runtime.py")+" previous"
    output.write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("live",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    describe(read(args.live),args.output)
