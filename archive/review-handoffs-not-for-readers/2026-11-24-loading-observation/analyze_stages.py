"""Replay the timestamped markers from the frozen post-window log span."""
import collections
import json
from pathlib import Path
import sys


def clock_us(text):
    minute,second,ms,us=map(int,text.split("."))
    assert 0<=second<60 and 0<=ms<1000 and 0<=us<1000
    return (((minute*60+second)*1000+ms)*1000+us)


def compute(root):
    read=lambda name:json.loads((root/name).read_text())
    window=read("LOADER_MARKERS.json")["files"]["server.stderr.log"]
    raw=read("POSTWINDOW_MARKERS.json")["files"]["server.stderr.log"]
    nextcheck=read("HANDOFF_CHECK.json")
    assert raw["from"]==window["last_offset"]==window["first_offset"] and raw["to"]>raw["from"]
    assert nextcheck["health_get"]["ready"] is True and read("READINESS_END.json")["ready"] is False
    assert nextcheck["separate_log_range"]["same_identity_after"] and nextcheck["separate_log_range"]["same_identity_before"]
    events=raw["markers"]
    assert all(raw["from"]<=r["offset"]<r["end_offset"]<=raw["to"] for r in events)
    assert all(a["end_offset"]<=b["offset"] for a,b in zip(events,events[1:]))
    grouped=collections.defaultdict(list)
    for e in events:
        grouped[e["marker"]].append(e)
    ordered=("layout_begin","layout_complete","host_registration_begin","host_registration_end","threadpool_init","api_loaded","http_listening")
    assert all(len(grouped[k])==1 for k in ordered)
    t={k:clock_us(grouped[k][0]["log_clock_text"]) for k in ordered}
    assert list(t.values())==sorted(t.values())
    ready=grouped["tensor_packed"]
    n,work,weight_bytes=map(int,grouped["layout_complete"][0]["fields"])
    assert n==len({m["fields"][0] for m in ready})==len(ready)==80
    assert sum(int(m["fields"][1]) for m in ready)==work==19200
    assert all(t["layout_begin"]<clock_us(m["log_clock_text"])<t["layout_complete"] for m in ready)
    ranges,gib=grouped["host_registration_begin"][0]["fields"]
    return {"source":"common/log.cpp uses elapsed microseconds since logger start, printed as minutes.seconds.milliseconds.microseconds. These are not wall times.",
      "logger_elapsed_us":t,"cpu_layout_seconds":(t["layout_complete"]-t["layout_begin"])/1e6,
      "host_registration_seconds":(t["host_registration_end"]-t["host_registration_begin"])/1e6,
      "threadpool_to_loaded_seconds":(t["api_loaded"]-t["threadpool_init"])/1e6,
      "seconds_until_server_reports_model_loaded_from_logger_start":t["api_loaded"]/1e6,
      "layout_start_to_server_loaded_seconds":(t["api_loaded"]-t["layout_begin"])/1e6,
      "packed_tensors":n,"selected_expert_projections":work,"packed_bytes":weight_bytes,
      "host_registered_ranges":int(ranges),"host_registered_GiB_rounded_in_log":float(gib),
      "threadpool_init_reported_n_threads":int(grouped["threadpool_init"][0]["fields"][0]),
      "ready_check_http_at_handoff":nextcheck["health_get"]["http_status"],"ready_http_utc":nextcheck["health_get"]["utc"]}


if __name__=="__main__":
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    res=compute(root)
    dest=root/"POSTWINDOW_STAGES.json"
    if "--check" in sys.argv:
        assert res==json.loads(dest.read_text())
        print("Reproduced 80 tensor markers and 7 stage boundaries")
    else:
        dest.write_text(json.dumps(res,indent=2)+"\n")
