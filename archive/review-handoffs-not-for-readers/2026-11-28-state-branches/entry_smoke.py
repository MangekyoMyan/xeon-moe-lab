"""Open and close the published foreground command against a separate empty user catalog."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from lab_trial import R, E, OWNER, record, load


def main():
    work=R/"entry"
    work.mkdir(exist_ok=False);(work/"build").mkdir()
    for f in ("BUILD.json","libllama-server-impl.so"):
        shutil.copyfile(R/"build"/f,work/"build"/f)
    shutil.copyfile(R/"BEFORE.json",work/"BEFORE.json")
    env={**os.environ,"XEON_BRANCH_WORKSPACE":str(work)}
    exe=str(R/"branch")
    def call(*args):
        ret=subprocess.run([exe,"--json",*args],env=env,user=OWNER.st_uid,group=OWNER.st_gid,
                           text=True,capture_output=True,timeout=120)
        if ret.returncode!=0:
            raise RuntimeError((args,ret.stderr[:350]))
        return json.loads(ret.stdout)
    with (work/"terminal.log").open("x") as log:
        proc=subprocess.Popen([exe,"open"],env=env,stdout=log,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
        try:
            deadline=time.monotonic()+1800
            sock=work/"user-store/manager.sock"
            while proc.poll() is None and time.monotonic()<deadline:
                if sock.is_socket():
                    break
                time.sleep(2)
            else:
                raise RuntimeError(("foreground entry failed to become ready",proc.returncode))
            initial=call("list")
            parent=call("turn","--text","これは端末のテストです。OKだけ答えてください。","--max-tokens","8")
            mark=call("mark","terminal-P")
            fork=call("fork","terminal-A","--from","terminal-P")
            switched=call("checkout","terminal-A")
            assert mark["snapshot"]["id"]==fork["branch"]["head"] and switched["replayed"]==0
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            rc=proc.wait(timeout=1300)
            if rc!=0:
                raise RuntimeError(("foreground entry exit",rc))
    window,=work.glob("WINDOW_BRANCH-CLI-*.json")
    restored,=work.glob("RESTORED_BRANCH-CLI-*.json");status=load(restored)
    assert load(window)["restore_required"] is False and status["API_health"]=="ok"
    assert status["config_sha256"]==load(R/"BEFORE.json")["config_sha256"]
    record(E/"entry.json",{"initial_branch":initial["branches"][0]["name"],"turn":parent["content"],"committed":mark["snapshot"]["id"],
                           "fork":fork["branch"]["id"],"checkout_replayed":switched["replayed"],"exit":rc,
                           "ordinary_api_pid":status["pid"],"ordinary_config":status["config_sha256"],
                           "actual_user_directory_created":(R/"user-store").exists()})
    print("foreground entry, CLI commands, interrupt, and ordinary API restoration passed")


if __name__=="__main__":
    main()
