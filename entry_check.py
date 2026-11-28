"""Run the actual public foreground command and recover the committed ambiguous turn."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from trial import R,F,LAB,read,write
from validation import catalog


def main():
    work=R/"entry";store=work/"user-store";build=work/"build";build.mkdir(exist_ok=True)
    for f in ("BUILD.json","libllama-server-impl.so"):shutil.copyfile(R/"build"/f,build/f)
    shutil.copyfile(R/"BEFORE.json",work/"BEFORE.json")
    old=read(R/"evidence/uncertain.json");env={**os.environ,"XEON_BRANCH_WORKSPACE":str(work)}
    exe=str(F/"x9ram-branch");owner=LAB.stat()
    def cli(*args):
        ret=subprocess.run([exe,"--json",*args],env=env,user=owner.st_uid,group=owner.st_gid,capture_output=True,text=True,timeout=200)
        assert ret.returncode==0,(args,ret.stderr[:400])
        return json.loads(ret.stdout)
    with (work/"launcher.log").open("a") as log:
        proc=subprocess.Popen([exe,"open","--checkpoint-mode","auto","--checkpoint-interval-tokens","8"],env=env,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,stdin=subprocess.DEVNULL)
        try:
            end=time.monotonic()+1600
            while proc.poll() is None and time.monotonic()<end:
                if (store/"manager.sock").is_socket():break
                time.sleep(1)
            else:raise RuntimeError(("launcher failed",proc.poll()))
            initial=cli("status");p=initial["checkpoint_policy"];assert p["mode"]=="auto" and p["interval_tokens"]==8 and initial["mode"]=="ready"
            b=catalog(store)
            for k in ("id","tokens","pending","messages","generation","base","head"):assert b[k]==old["after"][k]
            initial_branch=cli("inspect","branch","main")
            assert initial_branch["last_replay_tokens"]==b["processed"]-old["before"]["processed"]
            assert all(not (store/"images"/name).exists() for name in old["unconfirmed_files_seen_by_test_only"])
            assert not p["outcomes"] and p["observations"]["prefill"]==1
            out=cli("turn","--text","先ほどこの枝に登録した検証番号を、半角の数字だけで答えてください。","--max-tokens","8")
            assert out["turn_committed"] and out["checkpoint"]["basis"]=="cold_fallback" and out["checkpoint"]["status"]=="saved"
            mark=cli("mark","recovered","--pin");assert mark["snapshot"]["id"]==out["checkpoint"]["snapshot_id"]
            final=cli("status");assert len(catalog(store)["messages"])==len(b["messages"])+2
        finally:
            if proc.poll() is None:proc.send_signal(signal.SIGINT)
            rc=proc.wait(timeout=1200)
            if rc!=0:raise RuntimeError(("launcher exit",rc))
    w=next(work.glob("RESTORED_BRANCH-CLI-*.json"));restored=read(w)
    assert restored["API_health"]=="ok" and restored["config_sha256"]==read(R/"BEFORE.json")["config_sha256"]
    write(R/"evidence/entry.json",{"args":["open","--checkpoint-mode","auto","--checkpoint-interval-tokens","8"],"initial":initial,
                                   "initial_branch":initial_branch,"recovered_equals_committed_tokens_and_pending":True,"turn":out,"mark":mark,
                                   "final":final,"ordinary_pid":restored["pid"],"answer_correct":out["content"].strip()=="731","exit":rc})
    print("Actual launcher restored the committed branch, replay",initial_branch["last_replay_tokens"],"answer",repr(out["content"]))

if __name__=="__main__":main()
