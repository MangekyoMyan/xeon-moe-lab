#!/usr/bin/env python3
"""Compare an unchanged Japanese question and deterministic generated answer."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from full_replica_experiment import LAB, MODEL, BASE_ENV, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=("B", "C"))
    args = parser.parse_args()
    assert "THP_enabled:\t0" in Path("/proc/self/status").read_text()
    out = LAB / ("results/optimization-q6-xeon-20261020/responses-" + args.candidate)
    out.mkdir(exist_ok=False)
    prompt = "次の質問に、日本語で300字程度、具体例を含めて答えてください。古い2CPUサーバーでLLMを使う場合、メモリ帯域とCPU間通信は応答速度にどう関係しますか？"
    (out/"prompt.txt").write_text(prompt)
    records = []
    for i, label in enumerate("A" + args.candidate*2 + "A"):
        d = out / f"{i}-{label}"
        d.mkdir()
        build = LAB / ("review-full-replica-20261020/build" if label == "A" else "review-q6-scales-20261020/build")
        env = dict(BASE_ENV, GGML_CPU_NUMA_FULL="2")
        if label == "C":
            env["GGML_CPU_Q6_STATIC_ROWS"] = "1"
        command = ["numactl","--physcpubind=0-23","--interleave=0,1",str(build/"bin/llama-completion"),
                   "-m",str(MODEL),"-t","24","-tb","24","-ngl","0","-c","4096","-b","512","-ub","512",
                   "--load-mode","mmap","--cpu-mask","0xffffff","--cpu-strict","1",
                   "--cpu-mask-batch","0xffffff","--cpu-strict-batch","1","--jinja","--offline",
                   "-f",str(out/"prompt.txt"),"--single-turn","--simple-io","--color","off",
                   "--no-display-prompt","--reasoning","off","--temp","0","--seed","42","-n","128","--perf"]
        print(json.dumps({"response":d.name}),flush=True)
        samples=[]; started=time.monotonic()
        with (d/"stdout.txt").open("w") as stdout, (d/"stderr.txt").open("w") as stderr:
            process = subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,stdin=subprocess.DEVNULL)
            try:
                while process.poll() is None:
                    s=snapshot(process.pid)
                    if s:samples.append(s)
                    if time.monotonic()-started>600:raise TimeoutError()
                    time.sleep(1)
            finally:
                if process.poll() is None:process.terminate()
                try:rc=process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();rc=process.wait()
        assert rc==0
        log=(d/"stderr.txt").read_text()
        timings={name:float(m.group(1)) for name in ("load","prompt eval","eval")
                 if (m:=re.search(r"common_perf_print:\s+"+name+r" time\s*=\s*([\d.]+) ms",log))}
        assert len(timings)==3
        n=int(re.search(r"common_perf_print:\s+eval time\s*=.*?/\s*(\d+) runs",log).group(1))
        records.append({"label":label,"response":(d/"stdout.txt").read_text(),
                        "timings_ms":timings,"eval_tokens":n,"command":command,
                        "environment":{k:v for k,v in env.items() if k.startswith(("GGML_","OMP_"))},
                        "max_disk_read_bytes":max(s["disk_read_bytes"] for s in samples),"samples":samples})
        report={"responses":records,"complete":len(records)==4,
                "responses_identical":len({r["response"] for r in records})==1}
        (out/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        assert report["responses_identical"]
        assert len({r["eval_tokens"] for r in records})==1
        print(json.dumps({"done":d.name,"tokens":n,"timings_ms":timings}),flush=True)


if __name__=="__main__":
    main()
