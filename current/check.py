#!/usr/bin/env python3
"""Inspect the recorded inference profile, its backing files, or the current CPU layout."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def processor_list(specification):
    values=set()
    for section in specification.split(","):
        bounds=[int(v) for v in section.split("-")]
        values.update(range(bounds[0],bounds[-1]+1))
    return sorted(values)

def topology():
    root=Path("/sys/devices/system/cpu")
    online=processor_list((root/"online").read_text().strip())
    groups={}
    allowed=set(os.sched_getaffinity(0))
    for cpu in online:
        place=root/f"cpu{cpu}"
        socket=int((place/"topology/physical_package_id").read_text())
        core=int((place/"topology/core_id").read_text())
        nodes=[int(p.name[4:]) for p in place.glob("node[0-9]*")]
        group=groups.setdefault((socket,core),{"socket":socket,"core":core,"nodes":sorted(nodes),"logical":[],"allowed":[]})
        group["logical"].append(cpu)
        if cpu in allowed:group["allowed"].append(cpu)
    representative=[v["allowed"][0] for _,v in sorted(groups.items()) if v["allowed"]]
    return {"physical_cores_online":len(groups),"logical_cpus_online":len(online),"affinity_of_this_process":sorted(allowed),
            "one_available_thread_per_core":representative,"physical_cpu_mask":hex(sum(1<<n for n in representative)),
            "cores":[v for _,v in sorted(groups.items())]}

def require(value,message):
    if not value: raise ValueError(message)

def check(profile,lab,live,full_weights,asset_root=ROOT):
    checked=[]
    def file(entry):
        p=lab/entry["file"]
        require(digest(p)==entry["sha256"],f"SHA256 differs: {p}")
        checked.append(str(p))
        return p
    config_file=file(profile["service_config"])
    config=read(config_file)
    for lib in profile["libraries"].values():file(lib)
    for asset in [profile["server_executable"],profile["plan"]]:file(asset)
    require(config["environment"]["GGML_CPU_IQ_PLAN"]==str(lab/profile["plan"]["file"]),"Plan path differs")
    plan=profile["plan"]
    require(plan["model_download_fingerprint"]==profile["model"]["download_receipt"]["sha256"],"Plan identifies another model receipt")
    published=asset_root/plan["published_copy"]
    require(digest(published)==plan["sha256"],"Packaged selection plan differs from the live plan")
    checked.append(str(published))
    for group in [profile["model"],profile["vision"]]:
        file(group["download_receipt"])
        for shard in group["files"]:
            path=lab/shard["file"]
            require(path.stat().st_size==shard["size"],f"Size differs: {path}")
            if full_weights:file(shard)
    require(config["model"]==str(lab/profile["model"]["first_shard"]),"Primary model path differs")
    require(config["vision_profile"]["projector"]==str(lab/profile["vision"]["files"][0]["file"]),"Projector path differs")
    result={"checks_passed":True,"content_hashed":checked,"model_and_vision_file_sizes_checked":True,
            "weight_contents_rehashed_in_this_call":full_weights}
    if live:
        pid=read(config_file.parent/"health.json")["pid"]
        proc=Path("/proc")/str(pid)
        rows=[line.split(None,5) for line in (proc/"maps").read_text().splitlines()]
        expected={str((lab/e["file"]).resolve()) for e in profile["libraries"].values()}
        names={Path(p).name for p in expected}
        mapped={r[5] for r in rows if len(r)==6 and Path(r[5].removesuffix(" (deleted)")).name in names}
        require(mapped==expected,"Running libraries differ from the manifest")
        for r in rows:
            if len(r)==6 and r[5] in expected:
                st=Path(r[5]).stat()
                require(st.st_ino==int(r[4]) and (os.major(st.st_dev),os.minor(st.st_dev))==tuple(int(v,16) for v in r[3].split(":")),"Mapped inode differs: "+r[5])
        env=dict(x.split("=",1) for x in (proc/"environ").read_bytes().decode().split("\0") if "=" in x)
        require(all(env.get(k)==v for k,v in config["environment"].items()),"Configured environment differs")
        require(all(k not in env for k in ["GGML_CPU_IQ_ROW_LOCAL","GGML_CPU_IQ_AUDIT_DIR","LD_PRELOAD","XEON_169_SHARED_PREFIX"]),"Additional experimental option is set")
        cmd=(proc/"cmdline").read_bytes().decode().rstrip("\0").split("\0")
        index=config["command"].index(str(lab/profile["server_executable"]["file"]))
        require(cmd==config["command"][index:] and (proc/"exe").resolve()==(lab/profile["server_executable"]["file"]).resolve(),"Running server differs")
        require(read(config_file.parent/"health.json")["pid"]==pid,"API restarted during inspection")
        result["verified_api_pid"]=pid
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["show","verify","topology"],nargs="?",default="show")
    parser.add_argument("--lab-root",type=Path,default=Path("/home/codex/projects/llm-lab"))
    parser.add_argument("--live",action="store_true",help="Also check the running API's mapped libraries and configuration on x9ram")
    parser.add_argument("--full-weights",action="store_true",help="Read and hash every GGUF including the projector")
    args=parser.parse_args()
    if (args.live or args.full_weights) and args.action!="verify":parser.error("use --live or --full-weights with verify")
    if args.action=="topology":
        result=topology()
    else:
        profile=read(ROOT/"runtime.json")
        if args.action=="verify":result=check(profile,args.lab_root.resolve(),args.live,args.full_weights)
        else:
            result={k:profile[k] for k in ("profile","hardware","inference","libraries","ld_library_path","plan","service","row_local","planned_cpu_upgrade")}
            result["model"]={k:profile["model"][k] for k in ("repository","revision","quantization","first_shard")}
            result["model"]["shard_count"]=len(profile["model"]["files"])
            result["vision"]={k:profile["vision"][k] for k in ("repository","revision")}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    try:main()
    except (OSError,KeyError,ValueError) as err:
        sys.exit(f"Verification could not complete: {err}")
