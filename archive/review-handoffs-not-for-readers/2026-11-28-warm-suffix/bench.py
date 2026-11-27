"""ABBA on four independent ordinary llama-server processes with one slot."""
import argparse
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from build import R,LAB,D,read,save,sha

prior=LAB/"normal24-timeline-20261124"
sys.path.insert(0,str(prior))
import study
import prefill
study.R=R
PERIODS=(("A1",64,False),("B1",128,False),("B2",128,True),("A2",64,True))
MODEL="deepseek-v4.1-flash-1017-mix256"


def phase(folder,name):
    if not folder.is_dir():return
    tmp=folder/"phase.new";tmp.write_text(name+"\n");tmp.replace(folder/"phase")


def request(run,label,kind,prompt,expected,cache):
    observer=run/"trace"
    actual=prompt if kind=="native" else None
    if kind=="native":
        endpoint="/completion"
        body={"prompt":prompt,"n_predict":1,"cache_prompt":cache,"temperature":0,"seed":19,"return_tokens":True,"id_slot":0}
    else:
        endpoint="/v1/chat/completions"
        body={"model":MODEL,"messages":prompt,"temperature":0,"seed":19,"max_tokens":48,"stream":True,
              "stream_options":{"include_usage":True},"cache_prompt":cache,"chat_template_kwargs":{"thinking_mode":"chat"}}
    data=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode()
    headers={"Authorization":"Bearer "+study.adopt.KEY.read_text().strip(),"Content-Type":"application/json"}
    con=http.client.HTTPConnection("127.0.0.1",8000,timeout=650)
    events=[];first=None;first_token=None;done=None;tokens=None;usage=None;timings=None;reason=None
    phase(observer,label)
    try:
        begin=time.monotonic_ns();con.request("POST",endpoint,data,headers)
        res=con.getresponse();header_ns=time.monotonic_ns()
        if res.status!=200:raise RuntimeError((label,res.status,res.read(1024)[:240]))
        if kind=="native":
            v=json.loads(res.read(4<<20));eof=time.monotonic_ns();timings=v["timings"];content=v.get("content","")
            tokens=v.get("tokens");reason=v.get("stop_type");done=eof
        else:
            pieces=[]
            while line:=res.readline(2<<20):
                ts=time.monotonic_ns()
                if not line.startswith(b"data:"):continue
                value=line[5:].strip()
                if value==b"[DONE]":
                    done=ts;events.append({"ns":ts,"done":True});continue
                v=json.loads(value);assert "error" not in v, (label,v["error"])
                events.append({"ns":ts,"event":v})
                if v.get("usage"):usage=v["usage"]
                if v.get("timings"):timings=v["timings"]
                for c in v.get("choices",[]):
                    delta=c.get("delta",{});s=delta.get("content") or ""
                    if s:
                        pieces.append(s)
                        if first is None:first=ts
                    if (s or delta.get("reasoning_content")) and first_token is None:first_token=ts
                    if c.get("finish_reason"):reason=c["finish_reason"]
            eof=time.monotonic_ns();content="".join(pieces)
    finally:
        con.close();phase(observer,"idle")
    assert timings is not None and done is not None
    if events:save(run/(label+"-SSE.json"),events)
    row={"label":label,"endpoint":endpoint,"request_sha256":hashlib.sha256(data).hexdigest(),"body_bytes":len(data),
         "input_tokens_expected":expected,"start_ns":begin,"headers_ns":header_ns,"first_visible_ns":first,
         "first_text_or_reasoning_ns":first_token,"done_ns":done,"eof_ns":eof,"content":content,"tokens":tokens,"usage":usage,"timings":timings,"finish":reason}
    with (run/"requests.jsonl").open("a") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    if kind=="chat":
        assert first is not None and usage and usage["prompt_tokens"]==expected, (label,usage,expected)
        assert usage["prompt_tokens_details"]["cached_tokens"]==timings["cache_n"] and usage["completion_tokens"]==timings["predicted_n"]
    assert timings["cache_n"]+timings["prompt_n"]==expected,(label,timings,expected)
    return row


def period(label,threshold,observed,period_i):
    fixtures=read(R/"FIXTURES.json");design=read(R/"DESIGN.json")
    run=R/"runs"/label;run.mkdir();owner=LAB.stat();os.chown(run,owner.st_uid,owner.st_gid)
    cfg=prefill.cfg_for(threshold)
    if observed:
        trace=run/"trace";trace.mkdir();os.chown(trace,owner.st_uid,owner.st_gid);phase(trace,"idle")
        cfg["environment"].update(LD_PRELOAD=str(D/"observe.so"),XEON_SUFFIX_RECORD=str(trace))
    save(run/"server-config.json",cfg);(run/"server-config.json").chmod(0o600)
    save(run/"SPEC.json",{"period":label,"threshold":threshold,"observed":observed,"source_profile":study.NEW,"design_sha256":sha(R/"DESIGN.json")})
    with (run/"supervisor.log").open("x") as log:
        proc=subprocess.Popen([sys.executable,"-B",str(R/"bench.py"),"serve",label],start_new_session=True,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+7200
            while proc.poll() is None and time.monotonic()<deadline:
                study.adopt.monitor()
                if (run/"health.json").is_file():
                    h=read(run/"health.json");pid=h.get("pid")
                    if pid and h["status"]=="running" and prefill.owns_listener(pid):
                        ticks=prefill.child_identity(proc.pid,pid,cfg)
                        try:code,body=prefill.get("/health")
                        except OSError:code,body=0,{}
                        if code==200 and body.get("status")=="ok":break
                time.sleep(3)
            else:raise RuntimeError(f"{label} startup failed, status {proc.poll()}")
            env=(Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
            assert all((k+"="+v).encode() in env for k,v in cfg["environment"].items())
            assert observed==any(v.startswith(b"LD_PRELOAD=") for v in env)
            backing=prefill.verify_backing(pid,read(R/"BEFORE.json")["mapped_runtime_hashes"])
            save(run/"READY.json",{"pid":pid,"start_ticks":ticks,"original_library_backing":backing,"thresholds":[threshold]*2,"observer":observed})
            def ask(name,kind,payload,n,reuse):
                study.adopt.monitor();assert prefill.child_identity(proc.pid,pid,cfg)==ticks
                code,slots=prefill.get("/slots",True);assert code==200 and len(slots)==1 and not slots[0].get("is_processing"),"single slot busy"
                row=request(run,name,kind,payload,n,reuse)
                print(label,name,row["timings"]["cache_n"],row["timings"]["prompt_n"],round(row["timings"]["prompt_ms"]/1000,3),flush=True)
                return row
            ctr=fixtures["controls"]
            ask("warmup","native",ctr["warmup"],len(ctr["warmup"]),False)
            if observed:
                calls=read_lines(run/"trace/calls.jsonl")
                assert any(x.get("logits_file") and any(u["tokens"]==len(ctr["warmup"]) for u in x["ubatches"]) for x in calls),"observer smoke"
            ask("cold96","native",ctr["cold96"],96,False)
            order=design["case_order"][period_i]
            for name in order:
                case=fixtures["cases"][name];n=len(case["history_ids"])
                seed=ask(name+"-history","native",case["history_ids"],n,False)
                assert seed["timings"]["cache_n"]==0
                ask(name+"-follow","chat",case["messages"],n+len(case["new_ids"]),True)
                if name=="departments":
                    e=fixtures["edit"];ask("departments-edit","chat",e["messages"],n+len(e["new_ids"]),True)
            a=ctr["warm_control_history"];b=ctr["warm_control_suffix"]
            ask("longcontrol-history","native",a,len(a),False);ask("longcontrol-follow","native",a+b,len(a)+len(b),True)
        finally:
            if proc.poll() is None:proc.terminate()
            rc=proc.wait(timeout=130)
            h=read(run/"health.json") if (run/"health.json").exists() else {}
            save(run/"STOP.json",{"guard_exit":rc,"guard_state":h.get("status"),"child_rc":h.get("child_returncode")})
            assert rc==0 and h.get("status")=="stopped_by_request" and h.get("child_returncode")==0,(rc,h.get("status"))
    rows=read_lines(run/"requests.jsonl");assert len(rows)==11
    save(run/"CHECKED.json",{"complete":True,"requests":len(rows),"requests_sha256":sha(run/"requests.jsonl")})


def read_lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def design():
    pj=read(D/"BUILD.json");assert sha(D/"observe.so")==pj["binary_sha256"] and sha(D/"observe.cpp")==pj["source_sha256"]
    return {"base":"b938c1d99714104f7a61610c52a8732c99b5daa4","runtime_config":study.NEW,
            "fixture_sha256":sha(R/"FIXTURES.json"),"observer_sha256":sha(D/"observe.so"),
            "periods":[list(p) for p in PERIODS],"case_order":[["warehouse","departments","python"]]*2+[["warehouse","python","departments"]]*2,
            "protocol":"Four independent native server startups. A1/B1 plain uninstrumented, B2/A2 with read-only batch/device tracer and full-logit readback. Compare pairs separately. Same warmup, idle one-slot, cold native seed with one predicted token, then actual /v1/chat/completions streamed continuation. All four original inference libraries and model/vision/plan unchanged.",
            "clocks":"Client wall to first nonempty content and DONE, server prompt and decode times; diagnostic extra synchronization, graph enumeration and F32 write are confined to B2/A2. Cached/new counts taken from actual outgoing usage and timings. Suffix actual submitted IDs and prefill slices/dispatch measured in B2/A2.",
            "checks":"Warehouse 16388+75 with a known lookup answer; new Japanese numeric and Python return-value questions; separately retained edited Japanese follow-up; cold96 and warmed 1024+160 controls. Check prefix terminal full-vocabulary F32 across thresholds before attributing changes. No numerical equivalence or arbitrary latency threshold presumed, no automatic deployment."}


def main():
    args=argparse.ArgumentParser();args.add_argument("action",choices=["design","serve","run"]);args.add_argument("period",nargs="?")
    a=args.parse_args()
    if a.action=="design":
        p=R/"DESIGN.json";assert not p.exists();save(p,design());return
    if a.action=="serve":
        assert os.geteuid()==0 and a.period in {v[0] for v in PERIODS}
        prefill.serve(R/"runs"/a.period);return
    assert a.period is None and __debug__ and os.geteuid()==0 and read(R/"DESIGN.json")==design()
    existing=[p[0] for p in PERIODS if (R/"runs"/p[0]/"CHECKED.json").exists()]
    assert existing==[p[0] for p in PERIODS[:len(existing)]] and len(existing)<4
    def interrupted(signum,frame):raise KeyboardInterrupt(f"controller signal {signum}")
    for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,interrupted)
    with (prior/".study.lock").open() as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (R/"BEFORE.json").exists():
            subprocess.run([sys.executable,"-I",str(study.CHECK),str(R/"BEFORE.json"),"--expected",str(study.EXPECTED)],check=True)
        with study.maintenance("SUFFIX-"+str(time.time_ns())):
            for i,args in enumerate(PERIODS):
                if args[0] in existing:continue
                save(R/"STATE.json",{"stage":"ABBA running","period":args[0],"started_unix":time.time(),"default_config_sha256":study.NEW})
                period(*args,i)
    save(R/"MEASURED.json",{"periods":[p[0] for p in PERIODS],"complete":True,"finished_unix":time.time()})

if __name__=="__main__":main()
