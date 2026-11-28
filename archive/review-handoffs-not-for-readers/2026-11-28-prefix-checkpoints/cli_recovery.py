"""Open and close the installed shell entry point on the experiment's interrupted synthetic store."""
from pathlib import Path
import json
import os
import signal
import sqlite3
import subprocess
import time
R=Path(__file__).resolve().parent
E=R/"ring-evidence"
def run():
    workspace=R/"entry"
    assert (workspace/"user-store").resolve()==(R/"ring-store").resolve()
    env={**os.environ,"XEON_BRANCH_WORKSPACE":str(workspace)}
    owner=R.stat()
    def cli(*args):
        return subprocess.check_output([str(R/"branch"),*args],cwd=R,env=env,text=True,
                                       user=owner.st_uid,group=owner.st_gid,extra_groups=[],timeout=400)
    def js(*args):return json.loads(cli("--json",*args))
    log=workspace/"terminal.log"
    with log.open("x") as sink:
        proc=subprocess.Popen([str(R/"branch"),"open","--checkpoint-mode","fixed","--checkpoint-interval-tokens","127"],
            cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            until=time.monotonic()+1800
            while proc.poll() is None and time.monotonic()<until:
                if (R/"ring-store/manager.sock").is_socket() and "Branch CLI ready" in log.read_text():
                    break
                time.sleep(2)
            else:raise RuntimeError("CLI startup failed or timed out")
            old=json.loads((E/"cancel.json").read_bytes())
            status=js("list")
            db=sqlite3.connect("file:"+str(R/"ring-store/branches.sqlite3")+"?mode=ro",uri=True)
            tok,pending,gen,replay=db.execute("SELECT tokens,pending,gen,replay FROM branches WHERE name='main'").fetchone()
            db.close()
            from array import array
            assert list(array("i",tok))==old["committed_ids"] and list(array("i",pending))==old["pending_ids"] and gen==old["generation"]
            assert status["mode"]=="ready"
            run=next((workspace/"runs").glob("interactive-*"))
            ready=json.loads((run/"READY.json").read_bytes())
            (E/"recover.json").write_text(json.dumps(dict(pid=ready["pid"],replayed=replay,mode=status["mode"],generation=gen,
                    command="./branch open --checkpoint-mode fixed --checkpoint-interval-tokens 127"),indent=2)+"\n")
            jobs=json.loads((R/"workflow-evidence/workflow-fixtures.json").read_bytes())
            prompt=E/"fixed127-messages.json";prompt.write_text(json.dumps(jobs["A"])+"\n");prompt.chmod(0o644)
            v=js("turn","--messages",str(prompt),"--max-tokens","12")
            events=[(e["position"],e["phase"]) for e in v["checkpoint"]["events"] if e["status"]=="saved"]
            assert events==[(127,"prefill"),(254,"prefill"),(381,"decode")],events
            assert v["processed"]==v["normalized_input_count"]+v["sampled_count"]-1
            mark=js("mark","nongrid-127");assert mark["snapshot"]["processed"]%127
            later=js("turn","--text"," x"*180,"--max-tokens","2")
            more=[e["position"] for e in later["checkpoint"]["events"] if e["status"]=="saved"]
            assert 508 in more and all(p%127==0 for p in more),more
            preview=js("checkpoint","preview","--return-position","508","--empty-suffix")
            text=cli("checkpoint","status")
            (E/"fixed127.json").write_text(json.dumps(dict(first=v,mark=mark,second=later,preview=preview,status_text=text),
                                                     ensure_ascii=False,indent=2)+"\n")
        finally:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
            rc=proc.wait(timeout=700)
            if rc:raise RuntimeError("CLI guard did not restore original normal service")

