"""Run the continuation from one immutable, full native slot image at both thresholds."""
import argparse
import contextlib
import fcntl
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from build import R, LAB, D, read, save, sha
from bench import phase, request, prefill, study

SH = R/"shared"
IMAGES = SH/"images"
PERIODS = (("SA1",64),("SB1",128),("SB2",128),("SA2",64))
CASES = ("warehouse","departments","python")


def make_dir(path):
    path.mkdir(exist_ok=True)
    own = LAB.stat()
    os.chown(path,own.st_uid,own.st_gid)


def slot(folder, verb, filename, n):
    assert verb in {"save","restore"} and Path(filename).name == filename
    auth = {"Authorization":"Bearer "+study.adopt.KEY.read_text().strip(),"Content-Type":"application/json"}
    con = http.client.HTTPConnection("127.0.0.1",8000,timeout=240)
    try:
        t = time.monotonic_ns()
        con.request("POST","/slots/0?action="+verb,json.dumps({"filename":filename}),auth)
        res = con.getresponse()
        data = json.loads(res.read(4096))
        end = time.monotonic_ns()
    finally:
        con.close()
    assert res.status==200,(verb,filename,res.status,data)
    file = IMAGES/filename
    nkey,bytekey = ("n_saved","n_written") if verb=="save" else ("n_restored","n_read")
    assert data["id_slot"]==0 and data[nkey]==n and data[bytekey] == file.stat().st_size and file.is_file(),(verb,n,data,file.stat().st_size)
    record = {"verb":verb,"response":data,"begin_ns":t,"done_ns":end,"sha256":sha(file),"bytes":file.stat().st_size}
    with (folder/"slot.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")
    return record


@contextlib.contextmanager
def server(folder, threshold):
    make_dir(folder)
    trace = folder/"trace"
    make_dir(trace)
    phase(trace,"idle")
    cfg = prefill.cfg_for(threshold)
    cfg["environment"].update(LD_PRELOAD=str(D/"observe.so"),XEON_SUFFIX_RECORD=str(trace))
    assert "--slot-save-path" not in cfg["command"]
    cfg["command"] += ["--slot-save-path",str(IMAGES)]
    save(folder/"server-config.json",cfg)
    (folder/"server-config.json").chmod(0o600)
    with (folder/"supervisor.log").open("x") as log:
        proc = subprocess.Popen([sys.executable,"-B",str(R/"shared_state.py"),"serve",folder.name],
                                start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+1800
            while proc.poll() is None and time.monotonic()<deadline:
                study.adopt.monitor()
                if (folder/"health.json").exists():
                    h = read(folder/"health.json")
                    pid = h.get("pid")
                    if pid and h["status"]=="running" and prefill.owns_listener(pid):
                        ticks = prefill.child_identity(proc.pid,pid,cfg)
                        try:
                            status,body = prefill.get("/health")
                        except OSError:
                            status,body = 0,{}
                        if status==200 and body.get("status")=="ok":
                            break
                time.sleep(3)
            else:
                raise RuntimeError(("private state comparison startup",proc.poll()))
            backing = prefill.verify_backing(pid, read(R/"BEFORE.json")["mapped_runtime_hashes"])
            env = (Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
            assert all((k+"="+v).encode() in env for k,v in cfg["environment"].items())
            save(folder/"READY.json",{"pid":pid,"start_ticks":ticks,"thresholds":[threshold]*2,
                                      "observer":True,"original_library_backing":backing})
            def idle():
                study.adopt.monitor()
                assert prefill.child_identity(proc.pid,pid,cfg)==ticks
                status,slots = prefill.get("/slots",True)
                assert status==200 and len(slots)==1 and not slots[0].get("is_processing")
            yield idle
        finally:
            if proc.poll() is None:
                proc.terminate()
            rc = proc.wait(timeout=130)
            h = read(folder/"health.json") if (folder/"health.json").exists() else {}
            save(folder/"STOP.json",{"guard_exit":rc,"guard_state":h.get("status"),"child_rc":h.get("child_returncode")})
            assert rc==0 and h.get("status")=="stopped_by_request" and h.get("child_returncode")==0,(rc,h.get("status"))


def period(label, threshold):
    folder = R/"runs"/label
    f = read(R/"FIXTURES.json")
    order = CASES if label in ("SA1","SB1") else (CASES[0],CASES[2],CASES[1])
    with server(folder,threshold) as idle:
        def ask(stage,kind,data,total,cache):
            idle()
            result = request(folder,stage,kind,data,total,cache)
            print(label,stage,result["timings"]["cache_n"],result["timings"]["prompt_n"],round(result["timings"]["prompt_ms"]/1000,3),flush=True)
            return result
        warm = f["controls"]["warmup"]
        ask("warmup","native",warm,len(warm),False)
        if label=="SA1":
            images = {}
            for name in CASES:
                c = f["cases"][name]
                h = c["history_ids"]
                seed = ask(name+"-seed","native",h,len(h),False)
                assert seed["timings"]["cache_n"]==0
                fname = name+".slot"
                assert not (IMAGES/fname).exists()
                idle()
                out = slot(folder,"save",fname,len(h))
                (IMAGES/fname).chmod(0o440)
                images[name] = {"file":fname,"bytes":out["bytes"],"sha256":out["sha256"],"cached_tokens":len(h)}
                probe = ask(name+"-directprobe","native",h+c["new_ids"][:8],len(h)+8,True)
                assert probe["timings"]["cache_n"]==len(h)
            save(SH/"IMAGES.json",images)
        images = read(SH/"IMAGES.json")
        for name in order:
            c = f["cases"][name]
            h = c["history_ids"]
            n = len(h)
            im = images[name]
            idle()
            s = slot(folder,"restore",im["file"],n)
            assert s["sha256"]==im["sha256"]
            # Re-serialize the actual restored state before using it. Retain its digest even if
            # backend padding/metadata changes, then check an eight-token functional probe.
            tmpname = label+"-"+name+"-roundtrip.slot"
            idle()
            copy = slot(folder,"save",tmpname,n)
            if copy["sha256"]==im["sha256"]:
                (IMAGES/tmpname).unlink()
            probe = ask(name+"-probe","native",h+c["new_ids"][:8],n+8,True)
            assert probe["timings"]["cache_n"]==n
            idle()
            s = slot(folder,"restore",im["file"],n)
            assert s["sha256"]==im["sha256"]
            follow = ask(name+"-follow","chat",c["messages"],n+len(c["new_ids"]),True)
            assert follow["timings"]["cache_n"]==n and follow["timings"]["prompt_n"]==len(c["new_ids"])
    save(folder/"CHECKED.json",{"requests_sha256":sha(folder/"requests.jsonl"),"actions_sha256":sha(folder/"slot.jsonl")})


def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("action",choices=["design","run","serve"])
    ps.add_argument("period",nargs="?")
    a = ps.parse_args()
    if a.action=="design":
        SH.mkdir(exist_ok=True)
        conf = {"source_design":sha(R/"DESIGN.json"),"fixture_sha256":sha(R/"FIXTURES.json"),
                "observer_sha256":sha(D/"observe.so"),"source_sha256":sha(R/"shared_state.py"),
                "order":list(PERIODS),"cases":list(CASES),"probe":"first eight tokens of the actual new question, all CPU under either threshold",
                "method":"One full native serialized slot per synthetic history. The producer records terminal f32 and an eight-token direct probe. Every private native server restores exactly the same file, reserializes for a full-state SHA, evaluates the probe, restores again, then handles the identical OpenAI SSE follow-up. The restore/probe time is outside the reported continuation latency. All four periods observed. No automatic RAM checkpoints in any period."}
        path = SH/"DESIGN.json"
        assert not path.exists()
        save(path,conf)
        return
    assert os.geteuid()==0 and __debug__
    if a.action=="serve":
        assert a.period in dict(PERIODS)
        prefill.serve(R/"runs"/a.period)
        return
    assert read(R/"MEASURED.json")["complete"] and sha(R/"shared_state.py")==read(SH/"DESIGN.json")["source_sha256"]
    assert shutil.disk_usage(SH).free>12*1024**3
    make_dir(IMAGES)
    def interrupted(signum,frame):
        raise KeyboardInterrupt(signum)
    for s in (signal.SIGHUP,signal.SIGTERM):
        signal.signal(s,interrupted)
    with (LAB/"normal24-timeline-20261124/.study.lock").open() as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with study.maintenance("SUFFIX-SHARED-"+str(time.time_ns())):
            for label,threshold in PERIODS:
                if (R/"runs"/label/"CHECKED.json").exists():
                    continue
                assert not (R/"runs"/label).exists()
                save(SH/"STATE.json",{"period":label,"threshold":threshold,"started_unix":time.time()})
                period(label,threshold)
    save(SH/"MEASURED.json",{"complete":True,"finished_unix":time.time()})


if __name__=="__main__":
    main()
