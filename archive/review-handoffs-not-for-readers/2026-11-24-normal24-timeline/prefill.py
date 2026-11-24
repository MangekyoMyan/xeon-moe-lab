"""Bounded cold-prompt comparisons of the two configured offload thresholds on the native API."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from study import LAB,R,E,P,NEW,adopt,read,write,fresh,sha,guarded_child
from telemetry import start as start_telemetry

sys.path.insert(0,str(P))
from practical_api import child_identity,owns_listener,verify_backing,get

ROOT=R/"prefill"
FLAGS=("GGML_OP_OFFLOAD_MIN_BATCH","GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH")
POLICIES=(("A1",64),("L",32),("H",128),("A2",64))


def tokens():
    sets=read(R/"driver/CASES.json")["cases"]
    long=read(LAB/"parallel-pilot-20261117/replay/CASES.json")["cases"][1]
    assert long["case_id"]=="capacity_long_ja"
    out={}
    for n,ci,name in ((29,0,"JA29"),(48,0,"JA48"),(48,1,"PY48"),(96,0,"JA96"),(96,1,"PY96"),(272,0,"JA272")):
        out[name]=sets[ci]["prefix_ids"][:n]; assert len(out[name])==n and out[name][0]==0
    out["JA512"]=[0]+long["expected_token_ids"][:511]
    assert len(out["JA512"])==512
    return out


def cfg_for(threshold):
    assert sha(E/"server-config.json")==NEW
    cfg=read(E/"server-config.json")
    for key in FLAGS:
        assert cfg["environment"][key]=="64"
        cfg["environment"][key]=str(threshold)
    cmd=cfg["command"];i=cmd.index("--host")+1;assert cmd[i]==cfg["host"]==adopt.HOST
    cmd[i]=cfg["host"]="127.0.0.1"
    assert cfg["port"]==8000 and cmd[cmd.index("-t")+1]==cmd[cmd.index("-tb")+1]=="24"
    return cfg


def preflight():
    ROOT.mkdir()
    fresh(ROOT/"TOKENS.json",tokens())
    fresh(ROOT/"DESIGN.json",{"policies":POLICIES,"thresholds_changed_together":FLAGS,
       "order":"Each process: standard JA64 warmup; the same 7 prompts forward then reverse, 4 generated tokens each, direct loopback text completion.",
       "cache":"cache_prompt=false; require actual prompt_n==expected length and report actual cache_n for every request.",
       "criteria":"Compare TTFT and server prompt_ms per length against both 64 controls. Retain identical geometry cases as crosschecks. Separate first generated token IDs and text from timing; short outputs alone do not certify general numerical quality."})
    for _,t in POLICIES:cfg_for(t)


def native_request(ids):
    body={"prompt":ids,"n_predict":4,"cache_prompt":False,"temperature":0,"seed":19,"stream":True,
          "return_tokens":True,"ignore_eos":True,"id_slot":0}
    raw=json.dumps(body,separators=(",",":")).encode()
    headers={"Content-Type":"application/json","Authorization":"Bearer "+adopt.KEY.read_text().strip()}
    con=http.client.HTTPConnection("127.0.0.1",8000,timeout=150)
    timing=None;chunks=[];first_text=None;first_token=None;response_ready=None
    try:
        start=time.perf_counter_ns()
        con.request("POST","/completion",raw,headers)
        resp=con.getresponse();response_ready=time.perf_counter_ns()
        assert resp.status==200
        while line:=resp.readline(262144):
            t=time.perf_counter_ns()
            if not line.startswith(b"data:"):continue
            s=line[5:].strip()
            if s==b"[DONE]":continue
            e=json.loads(s)
            assert "error" not in e
            if e.get("stop"):
                timing=e["timings"]
                stop_type=e.get("stop_type")
                break
            txt=e.get("content","");ids_out=e.get("tokens",[])
            if txt and first_text is None:first_text=t
            if ids_out and first_token is None:first_token=t
            chunks.append({"ns":t,"tokens":ids_out,"text":txt})
        end=time.perf_counter_ns()
    finally:
        con.close()
    assert timing and timing["cache_n"]==0 and timing["prompt_n"]==len(ids) and timing["predicted_n"]==4
    returned=[x for e in chunks for x in e["tokens"]]
    assert len(returned)==4 and first_token is not None
    return dict(start_ns=start,headers_ns=response_ready,first_token_ns=first_token,first_nonempty_content_ns=first_text,
                final_event_ns=end,server=timing,output_ids=returned,text="".join(x["text"] for x in chunks),chunks=chunks,
                stop_type=stop_type,request_bytes=len(raw),cache_prompt=False)


def serve(run):
    # The regular native guard reaps its one child on this controller's SIGTERM.
    try:guarded_child(run)
    except KeyboardInterrupt:
        h=read(run/"health.json")
        if h.get("status")!="stopped_by_request":raise


def run_one(label,t):
    cfg=cfg_for(t);run=R/"runs"/("pp-"+label);run.mkdir()
    owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    fresh(run/"server-config.json",cfg);(run/"server-config.json").chmod(0o600)
    fresh(run/"SPEC.json",{"tag":label,"threshold":t,"base":NEW,"loopback_only":True,
                           "public_environment_delta":{k:str(t) for k in FLAGS}})
    stop,workers=start_telemetry(run,Path(cfg["command"][5]),False)
    try:
        with (run/"supervisor.log").open("x") as log:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"serve",label],start_new_session=True,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            try:
                until=time.monotonic()+7200;ready=None
                while proc.poll() is None and time.monotonic()<until:
                    adopt.monitor()
                    if (run/"health.json").is_file():
                        h=read(run/"health.json");pid=h.get("pid")
                        if pid and h["status"]=="running" and (Path("/proc")/str(pid)/"exe").resolve()==E/"build/bin/llama-server" and owns_listener(pid):
                            ticks=child_identity(proc.pid,pid,cfg)
                            try:code,b=get("/health")
                            except OSError:code,b=0,{}
                            if code==200 and b.get("status")=="ok":
                                ready={"pid":pid,"start_ticks":ticks,"supervisor":proc.pid,"socket_inode":owns_listener(pid)};break
                    time.sleep(4)
                assert ready is not None,"native startup failed or timed out"
                env=(Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
                assert all((k+"="+v).encode() in env for k,v in cfg["environment"].items())
                ready["backing"]=verify_backing(pid,read(R/"BEFORE.json")["mapped_runtime_hashes"]);fresh(run/"READY.json",ready)
                all_tokens=read(ROOT/"TOKENS.json")
                requests=[("warmup","JA64",all_tokens["JA272"][:64])]
                for rep,order in enumerate((list(all_tokens),list(reversed(all_tokens))),1):
                    requests.extend((rep,n,all_tokens[n]) for n in order)
                with (run/"requests.jsonl").open("x",buffering=1) as f:
                    for repeat,name,ids in requests:
                        adopt.monitor()
                        assert child_identity(proc.pid,pid,cfg)==ticks and owns_listener(pid)==ready["socket_inode"]
                        code,slots=get("/slots",True);assert code==200 and all(x.get("is_processing") is False for x in slots)
                        row=native_request(ids)
                        f.write(json.dumps({"rep":repeat,"fixture":name,"input_len":len(ids),**row},ensure_ascii=False,separators=(",",":"))+"\n")
                print("prefill",label,"14 timed HTTP requests complete",flush=True)
            finally:
                if proc.poll() is None:proc.terminate()
                rc=proc.wait(timeout=100)
                fresh(run/"STOP.json",{"supervisor_exit":rc,"guard_status":read(run/"health.json")["status"] if (run/"health.json").exists() else None})
                assert rc==0
    finally:
        stop.set()
        for w in workers:w.join(20)


def run_all():
    assert read(ROOT/"TOKENS.json")==tokens() and read(ROOT/"DESIGN.json")["policies"]==[list(x) for x in POLICIES]
    for label,t in POLICIES:
        write(ROOT/"STATE.json",{"running":label,"threshold":t})
        run_one(label,t)
    write(ROOT/"STATE.json",{"status":"four native processes complete"})


if __name__=="__main__":
    assert len(sys.argv)==3 and sys.argv[1]=="serve" and sys.argv[2] in dict(POLICIES) and os.geteuid()==0
    serve(R/"runs"/("pp-"+sys.argv[2]))
