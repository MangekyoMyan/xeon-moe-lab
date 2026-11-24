"""Read only the small, bounded byte ranges for this launch's log bookmarks."""
import argparse
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parent/"deepseek41-bringup-20261101"
PATTERNS={
    "layout_begin":rb"IQ cache: preparing CPU single-token layout",
    "tensor_packed":rb"IQ plan: ready (blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight); experts (\d+)/(\d+); total ([0-9.]+) GiB",
    "layout_complete":rb"IQ plan: complete (\d+) tensors, (\d+) selected expert projections, (\d+) bytes",
    "host_registration_begin":rb"expert DMA: registering (\d+) ranges, ([0-9.]+) GiB",
    "host_registration_end":rb"expert DMA: registration complete",
    "threadpool_init":rb"llama threadpool init, n_threads = (\d+)",
    "vision_tensors":rb"load_tensors: loaded (\d+) tensors from ",
    "projector_warmup":rb"get_dummy_batch: warmup with image size",
    "http_listening":rb"server is listening|HTTP server is listening|srv\s+server_on_li\w*:\s+listening on",
    "starting_model":rb"srv\s+load_model: loading model",
    "api_loaded":rb"main: model loaded|srv\s+(?:load_model|server_run_i\w*):\s+model loaded",
    "startup_capability":rb"clip_model_loader: has vision encoder",
    "gguf_header":rb"llama_model_loader: loaded meta data",
    "http_ready_marker":rb"srv\s+update_slots: all slots are idle",
}


def markers(blob,base,initial_partial=False):
    found=[]
    for n,line in enumerate(blob.splitlines(keepends=True)):
        start=base
        base+=len(line)
        if initial_partial and n==0:
            continue
        line=re.sub(rb"\x1b\[[0-9;]*m",b"",line)
        for k,pattern in PATTERNS.items():
            m=re.search(pattern,line)
            if m:
                tm=re.match(rb"([0-9.]{6,25})\s+[A-Z]\s",line)
                found.append(dict(marker=k,offset=start,end_offset=base,
                    log_clock_text=tm[1].decode() if tm else None,fields=[v.decode("ascii") for v in m.groups()]))
                break
    return found


def main():
    start=json.loads((ROOT/"WINDOW_START.json").read_text())
    checkpoints=[{"monotonic":start["monotonic"],"wall":start["wall"],"identity_matches":True,"logs":start["logs"]}]
    checkpoints.extend(json.loads(l) for l in (ROOT/"LOG_OFFSETS.jsonl").read_text().splitlines())
    result={}
    for name in ("server.stdout.log","server.stderr.log"):
        expected=start["logs"][name]
        items=[c["logs"][name] for c in checkpoints]
        continuous=all(c["identity_matches"] for c in checkpoints) and all(
            x.get("fd_matches",True) and (x.get("dev"),x.get("ino"))==(expected["dev"],expected["ino"]) for x in items
        ) and all(b["size"]>=a["size"] for a,b in zip(items,items[1:]))
        out={"path_basename":name,"observed_continuous":continuous,"first_offset":expected["size"],
             "last_offset":items[-1].get("size"),"dev":expected["dev"],"ino":expected["ino"]}
        result[name]=out
        if not continuous:
            continue
        p=ENGINE/name
        with p.open("rb") as f:
            st=os.fstat(f.fileno())
            assert (st.st_dev,st.st_ino)==(expected["dev"],expected["ino"]) and st.st_size>=out["last_offset"]
            n=out["last_offset"]-out["first_offset"]
            out["bytes_added"]=n
            if n>2**20:
                out["in_window_scan"]="over 1 MiB bound; raw range not read"
                continue
            near=min(16384,out["first_offset"])
            prev=os.pread(f.fileno(),near,out["first_offset"]-near)
            data=os.pread(f.fileno(),n,out["first_offset"])
            assert len(data)==n and len(prev)==near
        out["in_window_markers"]=markers(data,out["first_offset"],prev!=b"" and not prev.endswith(b"\n"))
        out["new_line_feeds"]=data.count(b"\n")
        out["prewindow_tail_markers_unattributed"]=markers(prev,out["first_offset"]-near,out["first_offset"]>near)
        for mark in out["in_window_markers"]:
            for i in range(1,len(items)):
                if items[i]["size"]>=mark["end_offset"]:
                    mark["first_observed_between_unix"]=[checkpoints[i-1]["wall"],checkpoints[i]["wall"]]
                    break
    print(json.dumps({"clock":"OS observation brackets; pre-window records have no captured launch-start offset",
                      "start_wall":checkpoints[0]["wall"],"end_wall":checkpoints[-1]["wall"],"files":result},indent=2))


def postwindow():
    from time import time
    start=json.loads((ROOT/"WINDOW_START.json").read_text())
    last=json.loads((ROOT/"LOG_OFFSETS.jsonl").read_text().splitlines()[-1])
    pid=start["model"]["pid"]
    proc=Path("/proc")/str(pid)
    def same_pid():
        try:
            return (Path("/proc/sys/kernel/random/boot_id").read_text().strip()==start["boot"] and
                    int((proc/"stat").read_text().rsplit(") ",1)[-1].split()[19])==start["model"]["start_ticks"])
        except (FileNotFoundError,ProcessLookupError):
            return False
    out={"wall":time(),"since_wall":last["wall"],"same_identity_before":same_pid(),"files":{}}
    if out["same_identity_before"]:
        for name in ("server.stderr.log","server.stdout.log"):
            ref=last["logs"][name]
            inherited=(proc/"fd"/str(start["logs"][name]["fd"])).stat()
            with (ENGINE/name).open("rb") as f:
                st=os.fstat(f.fileno())
                a,b=ref["size"],st.st_size
                ok=(st.st_dev,st.st_ino)==(ref["dev"],ref["ino"])==(inherited.st_dev,inherited.st_ino) and 0<=b-a<=2**20
                v={"first_offset":a,"last_offset":b,"fd_path_identity_matched_and_range_bounded":ok,"markers":[]}
                if ok:
                    dat=os.pread(f.fileno(),b-a,a)
                    assert len(dat)==b-a
                    mid_line=bool(a) and os.pread(f.fileno(),1,a-1)!=b"\n"
                    v.update(bytes_added=len(dat),markers=markers(dat,a,mid_line))
                out["files"][name]=v
    out["same_identity_after"]=same_pid()
    print(json.dumps(out))


def reparse_handoff():
    saved=json.loads((ROOT/"HANDOFF_CHECK.json").read_text())["separate_log_range"]
    original=json.loads((ROOT/"WINDOW_START.json").read_text())
    out={}
    for name in ("server.stdout.log","server.stderr.log"):
        v=saved["files"][name]
        assert saved["same_identity_before"] and saved["same_identity_after"] and v["fd_path_identity_matched_and_range_bounded"]
        a,b=v["first_offset"],v["last_offset"]
        ref=original["logs"][name]
        with (ENGINE/name).open("rb") as f:
            st=os.fstat(f.fileno())
            assert (st.st_dev,st.st_ino)==(ref["dev"],ref["ino"]) and st.st_size>=b and 0<=b-a<=2**20
            raw=os.pread(f.fileno(),b-a,a)
            assert len(raw)==b-a
            mid=bool(a) and os.pread(f.fileno(),1,a-1)!=b"\n"
        out[name]={"from":a,"to":b,"dev":st.st_dev,"ino":st.st_ino,"markers":markers(raw,a,mid)}
    print(json.dumps({"scope":"Same saved handoff byte ranges decoded with the actual server function prefixes. No new health request.",
        "previous_offset_time_unix":saved["since_wall"],"original_handoff_time_unix":saved["wall"],"files":out},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--post-window",action="store_true")
    ap.add_argument("--reparse-handoff",action="store_true")
    args=ap.parse_args()
    if args.reparse_handoff:
        reparse_handoff()
    elif args.post_window:
        postwindow()
    else:
        main()
