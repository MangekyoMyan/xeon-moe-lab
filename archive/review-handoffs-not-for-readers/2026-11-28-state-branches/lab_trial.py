"""Acquisition on x9ram. All request material in this script is synthetic."""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
F = R / "checkout/current/experiments/state-branches"
sys.path.insert(0, str(F))
from branchctl import rpc
from native import Native, Error, encode, digest, unpack
from x9ram_pilot import engine, manager, window, private, record, load, LAB

STORE = R / "test-store"
E = R / "evidence"
OWNER = LAB.stat()
PARENT = (
    "この会話は架空の保管台帳です。枝ごとに保管箱の番号を一つ登録します。"
    "後で番号を尋ねられたら、その枝で登録された半角の数字だけを返してください。"
    "まだ番号は登録していません。このメッセージには OK だけを返してください。"
)


def register(code):
    return (
        "この枝の保管箱番号は "
        + code
        + " です。後で参照するため覚えてください。今は半角の OK だけ返してください。"
    )


QUERY = "この枝で登録した保管箱番号を答えてください。半角の数字だけを返してください。"
LONG = (
    "次の行は関係のない梱包票です。箱番号の登録は変わりません。\n"
    + "\n".join(f"仮伝票 W-{i:03d} 梱包済み" for i in range(28))
    + "\n票を受領したら半角の OK だけ返信してください。"
)


def db_branch(name):
    con = sqlite3.connect(f"file:{STORE/'branches.sqlite3'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return dict(con.execute("SELECT * FROM branches WHERE name=?", (name,)).fetchone())
    finally:
        con.close()


def log(run, label, v):
    with (run / "observations.jsonl").open("a") as f:
        f.write(encode({"step": label, **v}) + "\n")


def ctl(run, label, *args, at=None, expect=None, store=STORE):
    cmd = [sys.executable, "-B", str(F / "branchctl.py"), "--store", str(store), "--json"]
    if at is not None:
        cmd += ["--at", encode(at)]
    cmd += list(args)
    start = time.monotonic_ns()
    z = subprocess.run(
        cmd, user=OWNER.st_uid, group=OWNER.st_gid, capture_output=True, text=True, timeout=650
    )
    finish = time.monotonic_ns()
    try:
        v = json.loads(z.stdout if z.returncode == 0 else z.stderr)
    except ValueError:
        raise RuntimeError((label, z.returncode, z.stderr[-700:]))
    log(
        run,
        label,
        {
            "argv": list(args),
            "returncode": z.returncode,
            "start_ns": start,
            "end_ns": finish,
            "response": v,
        },
    )
    if expect:
        assert z.returncode == 2 and v.get("code") == expect, (label, expect, v)
    else:
        assert z.returncode == 0, (label, v)
    return v


def keep_mark(run, label):
    info = ctl(run, "inspect-" + label, "inspect", "snapshot", label)
    assert info["image_status"] == "available"
    src = STORE / "images" / (info["image_sha256"] + ".slot")
    target = E / "references" / (label + ".slot")
    if not target.exists():
        shutil.copyfile(src, target)
    assert digest(target) == info["image_sha256"]
    return info


def phase(run, label):
    trace = run / "trace"
    if trace.is_dir():
        tmp = trace / "phase-new"
        tmp.write_text(label + "\n")
        tmp.replace(trace / "phase")


def basic(run, binding):
    with manager(STORE, binding, run / "manager.log"):
        ctl(run, "new-store", "status")
        p = ctl(run, "parent-turn", "turn", "--text", PARENT, "--max-tokens", "9")
        assert p["pending_sampled_ids"] and p["timings"]["prompt_n"] == p["normalized_input_count"]
        ctl(run, "parent-mark", "mark", "parent", "--pin")
        base = keep_mark(run, "parent")
        fork = ctl(run, "fork-A", "fork", "A", "--from", "parent")
        assert fork["branch"]["head"] == base["id"]
        ctl(run, "enter-A", "checkout", "A")
        a = ctl(run, "register-A", "turn", "--text", register("731"), "--max-tokens", "1")
        ctl(run, "mark-A", "mark", "point-A")
        pa = keep_mark(run, "point-A")
        back = ctl(run, "return-P", "checkout", "main")
        assert back["replayed"] == 0
        before = ctl(run, "before-fork-B", "status")
        fork = ctl(run, "fork-B", "fork", "B", "--from", "parent")
        after = ctl(run, "after-fork-B", "status")
        assert (
            after["image_bytes"] == before["image_bytes"] and fork["branch"]["head"] == base["id"]
        )
        ctl(run, "enter-B", "checkout", "B")
        b = ctl(run, "register-B", "turn", "--text", register("284"), "--max-tokens", "1")
        ctl(run, "mark-B", "mark", "point-B")
        pb = keep_mark(run, "point-B")
        assert pa["id"] != pb["id"] and pa["image_sha256"] != pb["image_sha256"]
        assert db_branch("A")["tokens"] != db_branch("B")["tokens"]
        assert a["processed"] == b["processed"], (a["processed"], b["processed"])
        long = ctl(run, "overwrite-window", "turn", "--text", LONG, "--max-tokens", "8")
        assert long["timings"]["prompt_n"] > 128
        switch = ctl(run, "back-to-A", "checkout", "A")
        assert switch["exact_head"] and switch["replayed"] == 0
        for label, obj in (("parent", base), ("point-A", pa), ("point-B", pb)):
            assert keep_mark(run, label)["image_sha256"] == obj["image_sha256"]
        ans = ctl(run, "answer-A", "turn", "--text", QUERY, "--max-tokens", "12")
        ctl(run, "final-A-mark", "mark", "answer-A")
        # Two simultaneous requests carry exactly the same ticket.
        stamp = rpc(STORE, {"op": "status"})["ticket"]

        def concurrent_switch(name):
            try:
                return {
                    "name": name,
                    "result": rpc(STORE, {"op": "checkout", "name": name, "ticket": stamp}),
                }
            except Error as e:
                return {"name": name, "error": e.code}

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            pair = list(pool.map(concurrent_switch, ["B", "main"]))
        assert (
            sum("result" in x for x in pair) == 1
            and sum(x.get("error") == "stale" for x in pair) == 1
        )
        log(run, "concurrent-checkout", {"same_ticket": stamp, "returns": pair})
        ctl(run, "return-A-after-race", "checkout", "A")
        ctl(run, "late-request", "mark", "late", at=stamp, expect="stale")
        # Release one reference while two others refer to the same parent image.
        ctl(run, "fork-disposable", "fork", "short-lived", "--from", "parent")
        ctl(run, "release-disposable", "release", "branch", "short-lived")
        ctl(run, "reject-pinned-release", "release", "mark", "parent", expect="pinned")
        ctl(run, "reject-small-budget", "budget", str(base["image_bytes"] - 1), expect="budget")
        assert keep_mark(run, "parent")["image_sha256"] == base["image_sha256"]
        current = ctl(run, "final-status", "status")
        record(
            E / "basic.json",
            {
                "P": base,
                "A": pa,
                "B": pb,
                "A_and_B_same_length": a["processed"],
                "B_appended": long["timings"]["prompt_n"],
                "A_answer": ans["content"],
                "A_correct": ans["content"].strip() == "731",
                "race": pair,
                "active": current["ticket"],
                "image_bytes": current["image_bytes"],
            },
        )


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("stage", choices=["basic", "deep", "numeric", "stock", "all-extra", "final"])
    pa.add_argument("--attempt", type=int, default=1)
    a = pa.parse_args()
    E.mkdir(exist_ok=True)
    (E / "references").mkdir(exist_ok=True)
    (R / "runs").mkdir(exist_ok=True)
    stages = ["deep", "numeric", "stock"] if a.stage == "all-extra" else [a.stage]
    with window(R, "PHASE1-" + a.stage):
        for stage in stages:
            run = R / "runs" / (stage + "-" + str(a.attempt))
            if run.exists():
                raise RuntimeError("choose a fresh attempt number")
            target_store = {"stock": R / "manual-store", "final": R / "perf-store"}.get(
                stage, STORE
            )
            with engine(
                R, target_store, run, observed=stage in ("numeric", "stock"), stock=stage == "stock"
            ) as binding:
                if stage == "basic":
                    basic(run, binding)
                elif stage == "final":
                    from final_cost import final_cost

                    final_cost(run, binding, target_store)
                else:
                    from more_trials import deep, numeric, stock

                    {"deep": deep, "numeric": numeric, "stock": stock}[stage](run, binding)
            record(run / "DONE.json", {"stage": stage, "verified": True})


if __name__ == "__main__":
    main()
