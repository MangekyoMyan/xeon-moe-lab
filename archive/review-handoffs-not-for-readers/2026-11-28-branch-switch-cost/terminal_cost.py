"""Time two real local CLI processes from checkout launch to the first reply byte."""
import os
from pathlib import Path
import subprocess
import sys
import time
from measure import action,server
from study_lib import R,STORE,LAB,FEATURE,Native,digest,read,reset_trial,write


def measure(run,binding,*,sizes=("short","long"),pairs=range(2),receipt="terminal.json"):
    native=Native(binding);owner=LAB.stat();results=[]
    for pair in pairs:
        for size in sizes:
            case=read(R/"cases"/(size+".json"))
            for version in (("base","new") if pair==0 else ("new","base")):
                code=R/"base-cli" if version=="base" else FEATURE
                reset_trial(native,case,False)
                with server(version,False,binding,run,"terminal",STORE):
                    prior=action(STORE,"turn",text=case["dirty_request"],max_tokens=1)
                    assert prior["processed"]==len(case["points"]["D"]["tokens"])
                    common=[sys.executable,"-B",str(code/"branchctl.py"),"--store",str(STORE)]
                    kw={"user":owner.st_uid,"group":owner.st_gid}
                    start=time.monotonic_ns()
                    switched=subprocess.run([*common,"--json","checkout","target"],capture_output=True,text=True,check=True,timeout=150,**kw)
                    installed=time.monotonic_ns()
                    import json
                    sw=json.loads(switched.stdout)
                    with subprocess.Popen([*common,"turn","--text",case["next_request"],"--max-tokens","8"],
                               stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,**kw) as process:
                        byte=process.stdout.read(1);seen=time.monotonic_ns()
                        more,err=process.communicate(timeout=150);done=time.monotonic_ns()
                        assert process.returncode==0 and byte and (byte+more).decode().strip()==case["answer"],err[:200]
                    assert sw["exact_head"] and sw["replayed"]==0
                    results.append({"size":size,"pair":pair,"version":version,"duplicate":False,"start_ns":start,"switch_process_exited_ns":installed,
                           "first_pipe_byte_ns":seen,"done_ns":done,"switch_ms":(installed-start)/1e6,"to_first_ms":(seen-start)/1e6,
                           "turn_alone_ms":(seen-installed)/1e6,"internal_switch_ms":sw["client_total_ms"],"answer":case["answer"],
                           "outgoing_snapshot_id":sw["outgoing_snapshot"],"replayed":sw["replayed"]})
    write(R/"evidence"/receipt,results)
