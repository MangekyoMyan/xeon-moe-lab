"""Allowlisted settings from the actual child, plus one maps read. No model file reads."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from readiness import ROOT, START, PROC, ident

ENGINE=ROOT.parent/"deepseek41-bringup-20261101"
OPTIONS=set(("-t -tb --threads --threads-batch --cpu-mask --cpu-mask-batch --cpu-strict --cpu-strict-batch "
             "-c -b -ub --parallel --mmproj --image-max-tokens --mtmd-batch-max-tokens "
             "--cache-type-k --cache-type-v --cache-ram --ctx-checkpoints -ngl "
             "--n-cpu-moe --load-mode --lazy-mode --poll --poll-batch --verbosity -m --physcpubind --interleave").split())
FLAGS=set(("--no-repack --no-context-shift --no-cont-batching --offline --cache-prompt --mmproj-offload "
           "--log-disable --log-timestamps --metrics").split())
ENV_KEYS=set(("OMP_NUM_THREADS OMP_DYNAMIC OMP_THREAD_LIMIT OMP_MAX_ACTIVE_LEVELS OMP_PROC_BIND OMP_PLACES "
              "GOMP_CPU_AFFINITY GGML_CPU_IQ3_COMPACT_MIB GGML_CPU_IQ_PLAN GGML_CUDA_REGISTER_HOST "
              "GGML_OP_OFFLOAD_MIN_BATCH GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH OPENBLAS_NUM_THREADS").split())


def selected(argv):
    out={}
    for i,arg in enumerate(argv):
        key,sep,attached=arg.partition("=")
        if key in OPTIONS:
            out[key]=attached if sep else argv[i+1]
        elif key in FLAGS:
            out[key]=True
    return out


def fingerprint(st):
    return dict(device_number=st.st_dev,inode=st.st_ino,size=st.st_size,
                mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)


def snap():
    assert ident()
    cutoff=8*1024**2
    file=ENGINE/"server-config.json"
    config_bytes=file.read_bytes()
    config=json.loads(config_bytes)
    argv=[b.decode(errors="replace") for b in (PROC/"cmdline").read_bytes().split(b"\0") if b]
    env={}
    for entry in (PROC/"environ").read_bytes().split(b"\0"):
        k,eq,v=entry.partition(b"=")
        if k.decode(errors="replace") in ENV_KEYS and eq:
            env[k.decode()]=v.decode(errors="replace")
    actual=selected(argv)
    settings=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic(),
                  pid=START["model"],boot_id=START["boot"],cmdline_selected=actual,
                  config_file_sha256=hashlib.sha256(config_bytes).hexdigest(),config_selected=selected(config["command"]),
                  environment_selected=env,environment_requested={k:v for k,v in config["environment"].items() if k in ENV_KEYS})
    assets=json.loads((ROOT.parent/"preupgrade-baseline-20261123/production-before/snapshot.json").read_text())["files"]["small"]
    wanted={e["path"]:e for e in assets if ".so." in Path(e["path"]).name or Path(e["path"]).name=="llama-server"}
    all_mapped={}
    ggufs={}
    # Collect only backing IDs and names; avoid saving process address ranges.
    for line in (PROC/"maps").read_text().splitlines():
        parts=line.split(None,5)
        if len(parts)!=6 or not parts[5].startswith("/"):
            continue
        path=parts[5]
        if not (path.endswith(".gguf") or path in wanted or any(s in Path(path).name for s in ("libggml-cpu.so","libggml-cuda.so","libggml-base.so","libllama.so"))):
            continue
        major,minor=(int(x,16) for x in parts[3].split(":"))
        m=dict(device_number=os.makedev(major,minor),inode=int(parts[4]))
        target=ggufs if path.endswith(".gguf") else all_mapped
        if path in target and target[path]!=m:
            raise RuntimeError("Backing file changed across mapping entries")
        target[path]=m
    mapped={}
    for path,observed in all_mapped.items():
        p=Path(path)
        disk=fingerprint(p.stat())
        ref=wanted.get(path)
        entry=dict(path=path,mapped=observed,on_disk=disk,mapped_matches_disk=all(disk[k]==v for k,v in observed.items()))
        if ref:
            entry["old_stat_match"]=all(ref["stat_before"][k]==v for k,v in disk.items())
            entry["old_sha256"]=ref["expected_sha256"]
        if entry["mapped_matches_disk"] and disk["size"]<=cutoff:
            with p.open("rb") as f:
                assert fingerprint(os.fstat(f.fileno()))==disk
                entry["sha256_read_now"]=hashlib.file_digest(f,"sha256").hexdigest()
                assert fingerprint(os.fstat(f.fileno()))==disk
            entry["hash_matches_old"]=bool(ref and entry["sha256_read_now"]==ref["expected_sha256"])
        else:
            entry["sha256_read_now"]=None
        mapped[p.name]=entry
    settings["project_binaries"]=mapped
    settings["gguf_mappings"]={Path(k).name:v for k,v in ggufs.items()}
    model=Path(actual["-m"])
    dev=model.stat().st_dev
    block=(Path("/sys/dev/block")/f"{os.major(dev)}:{os.minor(dev)}").resolve(strict=True)
    parent=block.parent if (block/"partition").is_file() else block
    settings["model_storage"]={"partition":block.name,"device":parent.name,
                              "rotational":int((parent/"queue/rotational").read_text()),"st_dev":dev}
    settings["runtime_leader_allowed"]={line.split(":")[0]:line.split(":",1)[1].strip()
           for line in (PROC/"status").read_text().splitlines() if line.startswith(("Cpus_allowed_list:","Mems_allowed_list:"))}
    settings["guard_script_sha256"]=hashlib.sha256((ENGINE/"supervise.py").read_bytes()).hexdigest()
    settings["same_identity_after"]=ident()
    assert settings["same_identity_after"]
    return settings


if __name__=="__main__":
    v=snap()
    with (ROOT/"CONFIG_OBSERVED.json").open("x") as f:
        json.dump(v,f,indent=2)
    print(v["cmdline_selected"],v["runtime_leader_allowed"])
    print({k:(x["mapped_matches_disk"],x.get("old_stat_match"),x.get("hash_matches_old")) for k,x in v["project_binaries"].items()})
