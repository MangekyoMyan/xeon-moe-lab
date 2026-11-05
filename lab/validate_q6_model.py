#!/usr/bin/env python3
"""Validate the Q6 kernel and static row assignment against the frozen model."""
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import full_replica_experiment as bench

LAB = Path(__file__).resolve().parent
ROOT = LAB / "review-q6-scales-20261020"
BASE = LAB / "review-full-replica-20261020"
OUT = LAB / "results/optimization-q6-xeon-20261020/validation"


def main():
    assert "THP_enabled:\t0" in Path("/proc/self/status").read_text()
    OUT.mkdir(exist_ok=False)
    report = {"complete": False, "checks": [], "logits": []}
    env = dict(bench.BASE_ENV, GGML_CPU_NUMA_FULL="2")
    def save():
        (OUT/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        for label, binary, options, extra in [
            ("quantize", ROOT/"build/bin/test-quantize-fns", [], {}),
            ("backend-kernel", ROOT/"build/bin/test-backend-ops",
             ["-b", "CPU", "-o", "MUL_MAT,MUL_MAT_ID", "-p", "type_a=q6_K"], {}),
            ("backend-static", ROOT/"build/bin/test-backend-ops",
             ["-b", "CPU", "-o", "MUL_MAT,MUL_MAT_ID", "-p", "type_a=q6_K"],
             {"GGML_CPU_Q6_STATIC_ROWS": "1"}),
        ]:
            command = ["numactl","--physcpubind=0-23","--interleave=0,1",str(binary),*options]
            print(json.dumps({"check": label}), flush=True)
            with (OUT/(label+".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               env=dict(env,**extra),check=True,timeout=900)
            check = {"label":label,"command":command,"passed":True}
            if label.startswith("backend-"):
                counts = re.findall(r"(\d+)/(\d+) tests passed", (OUT/(label+".log")).read_text())
                assert counts and all(int(a) == int(b) and int(b) > 0 for a,b in counts)
                check["test_counts"] = counts
            report["checks"].append(check)
            save()
        source = (LAB/"deepseek-logits-probe.cpp").read_text()
        source = source.replace("context_params.n_threads = 12;", "context_params.n_threads = 24;")
        source = source.replace("ggml_threadpool_params_default(12)", "ggml_threadpool_params_default(24)")
        (OUT/"logits-probe.cpp").write_text(source)
        for label, root, extra in (("A", BASE, {}), ("B", ROOT, {}),
                                   ("C", ROOT, {"GGML_CPU_Q6_STATIC_ROWS":"1"})):
            binary = OUT/(label+"-logits")
            command = ["c++","-std=c++17","-O2","-I"+str(root/"source/include"),
                       "-I"+str(root/"source/ggml/include"),str(OUT/"logits-probe.cpp"),
                       "-L"+str(root/"build/bin"),"-Wl,-rpath,"+str(root/"build/bin"),
                       "-lllama","-lggml-cpu","-lggml-base","-pthread","-o",str(binary)]
            subprocess.run(command,check=True)
            command = ["numactl","--physcpubind=0-23","--interleave=0,1",str(binary),
                       str(bench.MODEL),str(OUT/(label+".bin"))]
            print(json.dumps({"logits":label,"threads":24}),flush=True)
            with (OUT/(label+".stdout")).open("w") as stdout, (OUT/(label+".stderr")).open("w") as stderr:
                subprocess.run(command,stdout=stdout,stderr=stderr,env=dict(env,**extra),
                               check=True,timeout=900)
            raw=(OUT/(label+".bin")).read_bytes()
            vocab,records,prefix=struct.unpack_from("<iii",raw)
            assert (vocab,records,prefix)==(262144,34,127)
            assert len(raw)==12+160*4+records*(8+vocab*4)
            report["logits"].append({"label":label,"count":vocab*records,
                                    "sha256":hashlib.sha256(raw).hexdigest(),"command":command})
            save()
        report["logits_bit_identical"]=len({r["sha256"] for r in report["logits"]})==1
        assert report["logits_bit_identical"]
        report["complete"]=True
    finally:
        save()
    print(json.dumps({"complete":True,"all_logits_identical":True,"logits_per_comparison":262144*34}),flush=True)


if __name__=="__main__":
    main()
