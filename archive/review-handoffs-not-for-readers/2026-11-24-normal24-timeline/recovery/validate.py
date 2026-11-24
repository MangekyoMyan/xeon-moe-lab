"""One final maintenance window for the full-model candidate and separate prefill experiment."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from study import R,LAB,adopt,condition,fresh,read,write,maintenance,sha
import whole_runs
import prefill


def bandwidth_reference():
    assert adopt.unit()["ActiveState"]=="inactive"
    owner=LAB.stat()
    ids=condition()["process_cpu_ids"]; env={"PATH":"/usr/bin:/bin","LANG":"C","OMP_DYNAMIC":"FALSE","OMP_NUM_THREADS":"24",
        "OMP_THREAD_LIMIT":"24","GOMP_CPU_AFFINITY":" ".join(map(str,ids))}
    data={}
    sensors=[p for d in Path("/sys/class/hwmon").glob("hwmon*") if (d/"name").read_text().strip()=="coretemp" for p in d.glob("temp*_input")]
    assert len(sensors)==26
    for name,policy in (("interleaved","--interleave=0,1"),("local_first_touch","--localalloc")):
        adopt.monitor();cmd=["/usr/bin/numactl","--physcpubind="+",".join(map(str,ids)),policy,str(LAB/"read-bandwidth"),"24"]
        proc=subprocess.Popen(cmd,user=owner.st_uid,group=owner.st_gid,extra_groups=[],env=env,
                              stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+30;temps=[]
        try:
            while proc.poll() is None:
                temps.append(max(int(p.read_text()) for p in sensors)/1000)
                assert temps[-1]<60 and time.monotonic()<deadline,"bandwidth probe thermal/time limit"
                time.sleep(.05)
            out,err=proc.communicate()
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
        assert proc.returncode==0,("stream reference",proc.returncode,err)
        j=json.loads(out);assert j["threads"]==24 and sorted(j["cpus"])==ids and j["checksum_passed"] and len(j["gb_s"])==9
        data[name]={"command":cmd,"read":j,"peak_sampled_CPU_C":max(temps) if temps else None}
    fresh(R/"MEMORY_REFERENCE.json",{"binary_sha256":sha(LAB/"read-bandwidth"),"source_sha256":sha(LAB/"read_bandwidth.c"),
        "runs":data,"scope":"2 GiB checksummed contiguous AVX reads, 9 passes each. Software bytes/time; not a universal maximum or a per-process IMC measurement."})


def main():
    if sys.argv[1:]==["preflight"]:
        whole_runs.preflight();prefill.preflight();print("Whole-model F/S and native prefill study preflight complete");return
    assert sys.argv[1:]==["run"] and os.geteuid()==0 and __debug__
    with (R/".study.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (R/"OPERATOR_STATE.json").is_file() and (whole_runs.W/"DESIGN.json").is_file() and (prefill.ROOT/"DESIGN.json").is_file()
        def interrupted(sig,frame):raise KeyboardInterrupt(f"validation controller signal {sig}")
        for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,interrupted)
        try:
            with maintenance("VALIDATION"):
                bandwidth_reference()
                whole_runs.run_all()
                prefill.run_all()
            write(R/"VALIDATION_STATE.json",{"status":"complete; original normal 24/24 restored","unix":time.time()})
        except BaseException as exc:
            write(R/"VALIDATION_STATE.json",{"status":"incomplete","exception":repr(exc),"unix":time.time()});raise


if __name__=="__main__":main()
