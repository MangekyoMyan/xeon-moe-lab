"""Prepare the measured one-flag profile beside the results. Do not install it."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def main():
    results=json.loads((ROOT/"PRACTICAL_COMPARISON.json").read_text())
    assert results["same_response_text"] and results["same_input_tokens_and_cache"] and results["server_decode_tps_ratio"]>1
    base=ROOT.parent/"deepseek41-bringup-20261101/server-config.json"
    raw=base.read_bytes()
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert sha(raw)==json.loads((ROOT/"production-before-live.json").read_text())["config_sha256"]
    profile=json.loads(raw);args=profile["command"];i=args.index("-t")+1
    assert args.count("-t")==1 and args[i]=="12" and args[args.index("-tb")+1]=="24"
    assert args[args.index("--host")+1]==profile["host"]=="10.10.10.2"
    args[i]="24"
    candidate,n=re.subn(rb'("-t"\s*,\s*)"12"',rb'\g<1>"24"',raw)
    assert n==1 and json.loads(candidate)==profile
    private=ROOT/"profiles"
    private.mkdir(mode=0o700,exist_ok=False)
    dest=private/"normal-24-24.json"
    with os.fdopen(os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as f:
        f.write(candidate)
    record={"status":"prepared; not installed","normal_configuration_source":str(base),"normal_base_sha256":sha(raw),
            "prepared_file_local":str(dest),"prepared_file_sha256":sha(candidate),"production_bind_preserved":True,
            "candidate_measurement":{"request":"same one text prompt","measured_candidate_bind":"127.0.0.1","matched_generated_text":True,
                "server_decode_ratio":results["server_decode_tps_ratio"]},
            "json_patch":[{"op":"test","path":"/command/"+str(i-1),"value":"-t"},
                          {"op":"test","path":"/command/"+str(i),"value":"12"},
                          {"op":"replace","path":"/command/"+str(i),"value":"24"}]}
    (ROOT/"PROFILE_CANDIDATE.json").write_text(json.dumps(record,indent=2)+"\n")
    print(dest,sha(candidate))


if __name__=="__main__":
    main()
