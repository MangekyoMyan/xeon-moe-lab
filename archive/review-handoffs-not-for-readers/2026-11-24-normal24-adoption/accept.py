"""One normal-endpoint text request with the unchanged client; do not retry a partial stream."""
import json
import os
import subprocess
import sys
import time
from adopt import ROOT,LAB,ENGINE,REVIEW,PREVIOUS,KEY,NEW,HOST,PORT,read,save,digest,idle,identity,get,snapshot_health

CLIENT_SHA="1fd4259d44a9f5c96efc89ad810854fefdc04f22422e1fee115d02883503d08d"
REQUEST_SHA="8c7dd777627326870076b7b4ec19fca9db085f84bfd461a2321a78d3363bfc39"


def main():
    assert os.geteuid()==0 and __debug__
    after=read(ROOT/"AFTER_IDENTITY.json");who,free=idle(NEW)
    assert (who["pid"],who["start_ticks"])==(after["pid"],after["start_ticks"])
    route=json.loads(subprocess.check_output(["/usr/sbin/ip","-j","route","get",HOST]))
    assert len(route)==1 and route[0]["type"]=="local" and route[0]["dev"]=="lo"
    req=REVIEW/"requests/normal_stream_ja.json";client=REVIEW/"tools/capture_stream.py"
    assert digest(req.read_bytes())==REQUEST_SHA and digest(client.read_bytes())==CLIENT_SHA
    model=read(req)["model"];cfg=read(ENGINE/"server-config.json")
    code,models=get("/v1/models",True)
    assert code==200 and model==cfg["model_alias"] and model in {v["id"] for v in models["data"]}
    assert not (ROOT/"stream").exists()
    save("REQUEST_BEGIN.json",{"unix":time.time(),"identity":who,"request_sha256":REQUEST_SHA,"client_sha256":CLIENT_SHA,
        "served_model":model,"direct_local_route":True,"slots":free,"maximum_requests":1,"thread_sampler_attached":False})
    env={"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","XEON_ADOPT_STREAM_KEY":KEY.read_text().strip()}
    args=[sys.executable,"-I",str(client),"--url",f"http://{HOST}:{PORT}/v1/chat/completions","--request",str(req),
          "--out",str(ROOT/"stream"),"--label","adopted-normal-24-24","--api-key-env","XEON_ADOPT_STREAM_KEY"]
    owner=LAB.stat()
    res=subprocess.run(args,env=env,user=owner.st_uid,group=owner.st_gid,extra_groups=[],stdin=subprocess.DEVNULL,timeout=340)
    summary=read(ROOT/"stream/summary.json")
    assert res.returncode==0 and summary["status"]=="complete" and summary["http"]["status"]==200
    assert digest((ROOT/"stream/request.effective.json").read_bytes())==REQUEST_SHA
    assert digest((REVIEW/"tools/replay_stream.py").read_bytes())=="c5403d60d7475f8eae5dd77107493eb48b4ec70e5f6742c9443af590a753f5a8"
    replay=json.loads(subprocess.check_output([sys.executable,"-I",str(REVIEW/"tools/replay_stream.py"),str(ROOT/"stream")]))
    save("STREAM_REPLAY.json",replay)
    gold=(PREVIOUS/"pilot24/stream/content.txt").read_bytes()
    actual=(ROOT/"stream/content.txt").read_bytes()
    timings=[]
    for line in (ROOT/"stream/events.jsonl").read_text().splitlines():
        block=json.loads(line).get("json")
        if isinstance(block,dict) and "timings" in block:timings.append(block["timings"])
    assert len(timings)==1 and actual and replay["done_and_eof_observed"] and replay["finish_reason"]=="stop"
    assert identity(NEW)==who
    code,health=get("/health");assert code==200 and health["status"]=="ok"
    record={"finish_unix":time.time(),"pid":who["pid"],"boot":who["boot"],"start_ticks":who["start_ticks"],
        "http":200,"runtime_unchanged":True,"health_after_request":"ok","input_request_hash":REQUEST_SHA,
        "same_content_as_prior_pilot":actual==gold,"content_sha256":digest(actual),"prior_pilot_content_sha256":digest(gold),
        "ttft_content_ms":summary["ttft_content_receive_ms"],"eof_ms":summary["elapsed_from_request_start"]["eof_ms"],
        "reported_usage":replay["server_usage"],"server_timings":timings[0],"finish_reason":replay["finish_reason"],
        "content_events":replay["content_events"],"content_gaps_ms":replay["receive_event_gaps_ms"]["content"],
        "extra_inference_requests":0,"all_thread_sampling":False}
    save("HEALTH_AFTER_REQUEST.json",snapshot_health());save("ACCEPTANCE.json",record)
    print(json.dumps({k:record[k] for k in ("pid","same_content_as_prior_pilot","ttft_content_ms","eof_ms","server_timings","finish_reason")}))


if __name__=="__main__":main()
