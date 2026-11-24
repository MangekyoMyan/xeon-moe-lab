"""Independent, greedy OpenAI chat responses on cache-empty prompts before/after threshold changes."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shared import HERE, LAB, PRIOR, adopt, fresh, prefill, read, sha, study, window, write

CAPTURE = LAB / "results-review-20261124/tools/capture_stream.py"
CASES = HERE / "CHAT_FIXTURES.json"
ORDER = (("A1",64),("B1",128),("B2",128),("A2",64))
MODEL = "deepseek-v4.1-flash-1017-mix256"


def payload(messages):
    return {"model":MODEL, "messages":messages, "temperature":0, "seed":19, "max_tokens":160,
            "n":1, "stream":True, "stream_options":{"include_usage":True}, "cache_prompt":False}


def post(host, route, value):
    con = http.client.HTTPConnection(host, 8000, timeout=15)
    try:
        data = json.dumps(value, ensure_ascii=False).encode()
        con.request("POST", route, data, {"Content-Type":"application/json", "Authorization":"Bearer "+adopt.KEY.read_text().strip()})
        resp = con.getresponse()
        body = resp.read(1<<20)
        if resp.status != 200:
            raise RuntimeError((route, resp.status, body[:200]))
        return json.loads(body)
    finally:
        con.close()


def freeze():
    assert adopt.identity(adopt.NEW)["pid"] == read(HERE/"BEFORE.json")["pid"]
    source = read(CASES)
    docs = {}
    for case in source:
        req = payload(case["messages"])
        name = case["id"]
        rendered = post(adopt.HOST,"/apply-template",req)["prompt"]
        count = post(adopt.HOST,"/v1/chat/completions/input_tokens",req)["input_tokens"]
        raw = post(adopt.HOST,"/tokenize",{"content":rendered,"add_special":False,"parse_special":True})["tokens"]
        if len(raw) != count:
            raw = post(adopt.HOST,"/tokenize",{"content":rendered,"add_special":True,"parse_special":True})["tokens"]
        assert len(raw)==count and count>0
        docs[name] = {"prompt_tokens":count, "template":rendered, "ids":raw}
    warm = payload([{"role":"user","content":"Reply with just the digit 3."}])
    params = HERE/"requests";params.mkdir()
    for case in source:
        fresh(params/(case["id"]+".json"),payload(case["messages"]))
    fresh(params/"warmup.json",warm)
    fresh(HERE/"CHAT_DESIGN.json",{"periods":ORDER,"fixtures_sha256":sha(CASES),"capture_sha256":sha(CAPTURE),
        "prompts":docs,"order":"Forward, reversed, forward, reversed. One unscored chat warmup per process.",
        "scoring":"Exact JSON/numeric answers, Python/SQLite examples, Japanese mail factual and tone checklist. Score both frozen arms; report differing text and incompletes.",
        "ordinary_cap":160,"prompt_cache":False})
    print({k:v["prompt_tokens"] for k,v in docs.items()},flush=True)


def request(run, name, length):
    req = HERE/"requests"/(name+".json")
    out = run/name
    env = {"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","XEON_CHAT_KEY":adopt.KEY.read_text().strip()}
    cmd = [sys.executable,"-I",str(CAPTURE),"--url","http://127.0.0.1:8000/v1/chat/completions",
           "--request",str(req),"--out",str(out),"--label",run.name+"/"+name,
           "--api-key-env","XEON_CHAT_KEY","--max-seconds","180","--timeout","150"]
    res = subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=210)
    if res.returncode:
        raise RuntimeError(f"{run.name}/{name} capture failed: {res.stdout[-700:]}")
    summary = read(out/"summary.json")
    events = [json.loads(l) for l in (out/"events.jsonl").read_text().splitlines()]
    end = [e["json"] for e in events if "json" in e and e["json"].get("usage") and e["json"].get("timings")]
    assert summary["status"]=="complete" and summary["http"]["status"]==200 and len(end)==1
    q=end[0]; usage=q["usage"]; t=q["timings"]
    assert t["cache_n"]==usage["prompt_tokens_details"]["cached_tokens"]==0
    assert t["prompt_n"]==usage["prompt_tokens"] and (length is None or length==t["prompt_n"])
    assert t["predicted_n"]==usage["completion_tokens"] and 0<t["predicted_n"]<=160
    return {"case":name,"prompt":t["prompt_n"],"cache_n":t["cache_n"],"output":t["predicted_n"],"prompt_ms":t["prompt_ms"],
            "ttft_ms":summary.get("ttft_content_receive_ms"),"reasoning_first_ms":summary.get("first_reasoning_receive_ms"),
            "finish":summary["finish_reason"],"text":(out/"content.txt").read_text(),"response_sha256":sha(out/"response_body.bin")}


def run_one(label,threshold,index):
    d = HERE/"runs"/("chat-"+label)
    d.mkdir();os.chown(d,LAB.stat().st_uid,LAB.stat().st_gid)
    design=read(HERE/"CHAT_DESIGN.json")
    cfg=prefill.cfg_for(threshold)
    fresh(d/"server-config.json",cfg);(d/"server-config.json").chmod(0o600)
    fresh(d/"SPEC.json",{"threshold":threshold,"profile":adopt.NEW,"decode":24,"prefill":24,"loopback":True})
    stop,threads=prefill.start_telemetry(d,study.E/"build/bin/llama-server",False)
    try:
        with (d/"supervisor.log").open("x") as log:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),"serve",label],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+7200
                while proc.poll() is None and time.monotonic()<deadline:
                    adopt.monitor()
                    if (d/"health.json").is_file():
                        h=read(d/"health.json");pid=h.get("pid")
                        if pid and h["status"]=="running" and (Path("/proc")/str(pid)/"exe").resolve()==study.E/"build/bin/llama-server" and prefill.owns_listener(pid):
                            ticks=prefill.child_identity(proc.pid,pid,cfg)
                            try:code,b=prefill.get("/health")
                            except OSError:code,b=0,{}
                            if code==200 and b.get("status")=="ok":break
                    time.sleep(3)
                else:raise RuntimeError(f"test server failed to become ready ({proc.poll()})")
                env=(Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
                assert all((k+"="+v).encode() in env for k,v in cfg["environment"].items())
                inode=prefill.owns_listener(pid)
                fresh(d/"READY.json",{"pid":pid,"supervisor":proc.pid,"start_ticks":ticks,"listener_inode":inode,
                    "thresholds_verified_from_environ":threshold,"backing":prefill.verify_backing(pid,read(HERE/"BEFORE.json")["mapped_runtime_hashes"])})
                names=list(design["prompts"])
                if index%2:names.reverse()
                with (d/"results.jsonl").open("x",buffering=1) as f:
                    for name in ["warmup",*names]:
                        adopt.monitor()
                        assert prefill.child_identity(proc.pid,pid,cfg)==ticks and prefill.owns_listener(pid)==inode
                        row=request(d,name,design["prompts"].get(name,{}).get("prompt_tokens"))
                        f.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(label,threshold,"eight chat cases finished",flush=True)
            finally:
                if proc.poll() is None:proc.terminate()
                rc=proc.wait(timeout=100)
                h=read(d/"health.json")
                fresh(d/"STOP.json",{"supervisor_return":rc,"child_return":h.get("child_returncode"),"guard":h["status"]})
                assert rc==0 and h["status"]=="stopped_by_request" and h["child_returncode"]==0
    finally:
        stop.set()
        for thread in threads:thread.join(20)


if __name__=="__main__":
    if sys.argv[1]=="freeze":freeze()
    elif sys.argv[1]=="serve":
        assert sys.argv[2] in dict(ORDER) and os.geteuid()==0
        prefill.serve(HERE/"runs"/("chat-"+sys.argv[2]))
    elif sys.argv[1]=="run":
        design=read(HERE/"CHAT_DESIGN.json")
        assert design["periods"]==[list(p) for p in ORDER] and design["fixtures_sha256"]==sha(CASES) and design["capture_sha256"]==sha(CAPTURE)
        with window("CHAT"):
            for i,(label,threshold) in enumerate(ORDER):
                write(HERE/"CHAT_STATE.json",{"running":label,"threshold":threshold})
                run_one(label,threshold,i)
        write(HERE/"CHAT_STATE.json",{"status":"complete","restored":read(HERE/"RESTORED_CHAT.json")["pid"]})
    else:raise SystemExit("use freeze, serve or run")
