"""Summarize the single real production client A/B without equating SSE events with tokens."""
import hashlib
import json
import sys
from pathlib import Path
from analyze_stream_tasks import analyze


def read(p):
    return json.loads(p.read_bytes())


def get(root):
    pilot=root/"pilot24"
    v=analyze(pilot,root/"stream",root)
    normal=v.pop("old_normal_request");candidate=v.pop("new_normal_request")
    assert normal["effective_request_sha256"]==candidate["effective_request_sha256"]
    s0=normal["server_timings"];s1=candidate["server_timings"]
    meta=read(pilot/"READY.json")
    stop=read(pilot/"STOP.json")
    assert stop["final_guard_state"]=="stopped_by_request" and stop["supervisor_exit"]==0
    assert meta["configured_environment_matches"] and meta["slots_idle"]
    return {"normal_12_24":normal,"candidate_24_24":candidate,
       "same_response_text":normal["content_sha256"]==candidate["content_sha256"],
       "same_usage":normal["reported_usage"]==candidate["reported_usage"],
       "same_input_tokens_and_cache":(s0["prompt_n"],s0["cache_n"])==(s1["prompt_n"],s1["cache_n"]),
       "server_decode_tps_ratio":s1["predicted_per_second"]/s0["predicted_per_second"],
       "first_content_ms_difference":candidate["client_first_content_ms"]-normal["client_first_content_ms"],
       "server_prompt_ms_difference":s1["prompt_ms"]-s0["prompt_ms"],
       "fixed_profile_difference":read(pilot/"BASELINE.json")["effective_differences"],"candidate_scheduling":v,
       "candidate_ready":True,"normal_config_remains_default":True,
       "scope":"One real normal-text request per setting, matched capture client and sampling interval, separate model processes. Projector init follows -t in this server. No image inference measured."}


if __name__=="__main__":
    root=Path(sys.argv[1]);v=get(root);p=root/"PRACTICAL_COMPARISON.json"
    if "--check" in sys.argv:
        assert json.loads(json.dumps(v))==read(p)
        print("Practical SSE server timings and worker accounting reproduced")
    else:
        p.write_text(json.dumps(v,indent=2)+"\n")
