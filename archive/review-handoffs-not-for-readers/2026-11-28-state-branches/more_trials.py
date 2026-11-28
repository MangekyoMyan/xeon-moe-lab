"""Failure/restart, parity, raw-window and overhead trials for the private CLI."""

import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from lab_trial import (
    R,
    F,
    STORE,
    E,
    OWNER,
    QUERY,
    Native,
    Error,
    ctl,
    log,
    db_branch,
    keep_mark,
    phase,
    record,
    load,
    digest,
    rpc,
    manager,
    private,
)
from native import pack, unpack
from storage import Store


def infile(name, value):
    p = E / (name + ".json")
    record(p, value)
    return str(p)


def fit(native, messages, target, make):
    """Freeze an exact token length with the live chat template."""
    cache = {}

    def size(k):
        if k not in cache:
            candidate = messages + [{"role": "user", "content": make(k)}]
            cache[k] = (candidate, native.render(candidate))
        return len(cache[k][1])

    lo, hi = 0, target + 32
    while lo < hi:
        mid = (lo + hi) // 2
        if size(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    for k in range(max(0, lo - 3), lo + 4):
        if size(k) == target:
            return cache[k]
    raise ValueError(("fixture length unavailable", target, size(lo)))


def manual_stream(native, ids, maximum, emit=lambda _: None):
    start = time.monotonic_ns()
    first = None
    parts = []
    samples = []

    def consume(resp):
        nonlocal first
        final = None
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                continue
            v = json.loads(raw)
            if v.get("stop"):
                final = v
            else:
                samples.extend(v.get("tokens", []))
                if v.get("content"):
                    if first is None:
                        first = time.monotonic_ns()
                    parts.append(v["content"])
                    emit(v["content"])
        assert final is not None and final["timings"]["prompt_n"] + final["timings"][
            "cache_n"
        ] == len(ids)
        return {
            "content": "".join(parts),
            "stream_token_ids": samples,
            "first_ns": first,
            "start_ns": start,
            "finish_ns": time.monotonic_ns(),
            "timings": final["timings"],
            "processed": final["tokens_cached"],
        }

    return native.call(
        "POST",
        "/completion",
        {
            "id_slot": 0,
            "prompt": ids,
            "n_predict": maximum,
            "temperature": 0,
            "seed": 19,
            "cache_prompt": True,
            "return_tokens": True,
            "stream": True,
            "response_fields": [
                "stop",
                "content",
                "tokens",
                "tokens_cached",
                "stop_type",
                "timings",
            ],
        },
        consume,
    )


@contextlib.contextmanager
def unavailable_image(info, label, damaged):
    source = STORE / "images" / (info["image_sha256"] + ".slot")
    backup = STORE / ("test-preserved-" + label)
    assert not backup.exists() and digest(source) == info["image_sha256"]
    source.rename(backup)
    try:
        if damaged:
            # Same size, different digest. The manager must reject it before a native load.
            shutil.copyfile(backup, source)
            with source.open("r+b") as f:
                f.seek(-1, 2)
                v = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([v[0] ^ 1]))
        yield
    finally:
        os.replace(backup, source)
        assert digest(source) == info["image_sha256"]


def deep(run, binding):
    native = Native(binding)
    base = load(E / "basic.json")
    anchors = {}
    for label, length in (("odd", 255), ("even", 256)):
        make = (
            lambda k: "この架空の照合表の確認番号は619です。後で答えてください。区切り用の列は次の通りです。"
            + (" x" * k)
            + "\n今は OK とだけ答えてください。"
        )
        messages, ids = fit(native, [], length, make)
        anchors[label] = {"messages": messages, "ids": ids}
        infile("anchor-" + label, messages)
    with manager(STORE, binding, run / "manager.log", R / "fault_manager.py"):
        began = ctl(run, "after-whole-engine-restart", "status")
        assert began["mode"] == "ready" and began["ticket"]["branch"] == base["active"]["branch"]
        ctl(run, "restart-enter-B", "checkout", "B")
        result = ctl(run, "answer-B-after-restart", "turn", "--text", QUERY, "--max-tokens", "12")
        results = {"B_answer": result["content"], "B_correct": result["content"].strip() == "284"}
        ctl(run, "switch-main", "checkout", "main")
        for name, point, damaged in (
            ("missing-A", base["A"], False),
            ("damaged-B", base["B"], True),
        ):
            ctl(run, "fork-" + name, "fork", name, "--from", point["id"])
            with unavailable_image(point, name, damaged):
                v = ctl(run, name + "-checkout", "checkout", name)
                assert v["replayed"] > 0 and v["skipped"][0]["reason"] == (
                    "damaged" if damaged else "missing"
                )
                # A materialized save independently parses and checks the recovered native IDs.
                new = ctl(run, name + "-materialize", "mark", "after-" + name)["snapshot"]
                assert new["processed"] == point["processed"] and new["id"] != point["id"]
                results[name] = {"receipt": v, "new_snapshot": new}
            ctl(run, "return-from-" + name, "checkout", "main")
        for label, case in anchors.items():
            ctl(run, "fork-" + label, "fork", label, "--from", "parent")
            ctl(run, "enter-" + label, "checkout", label)
            v = ctl(
                run,
                "seed-" + label,
                "turn",
                "--messages",
                str(E / ("anchor-" + label + ".json")),
                "--max-tokens",
                "1",
            )
            assert v["processed"] == len(case["ids"]) and v["pending_sampled_ids"]
            ctl(run, "hold-" + label, "mark", label)
            case["snapshot"] = keep_mark(run, label)
            b = db_branch(label)
            assert unpack(b["tokens"]) == case["ids"]
            case["messages_after"] = json.loads(b["messages"])
        ctl(run, "before-native-errors", "checkout", "main")
        plan = run / "test-fault.json"
        for count in (1, 2):
            record(plan, {"mode": "absent-native-image", "count": count})
            err = ctl(
                run,
                "reject-native-" + str(count),
                "checkout",
                "odd",
                expect="native_error" if count == 1 else "rollback_failed",
            )
            status = ctl(run, "retained-head-" + str(count), "status")
            assert status["ticket"]["branch"] == db_branch("main")["id"]
            assert status["mode"] == ("ready" if count == 1 else "invalid")
            if count == 1:
                assert err["details"]["rollback"]["exact_head"]
            else:
                ctl(
                    run,
                    "blocked-invalid-slot",
                    "turn",
                    "--text",
                    "invalid",
                    "--max-tokens",
                    "1",
                    expect="invalid_slot",
                )
                v = ctl(run, "explicit-recovery", "recover")
                assert v["exact_head"] and v["replayed"] == 0
        ctl(run, "different-branches", "list")
    # A changed codec/model binding must reject the durable catalog before any restore.
    changed = dict(native.identity)
    changed["slot_format"] = "different-test-codec"
    bad = load(binding)
    bad["identity"] = changed
    bp = run / "incompatible-binding.json"
    record(bp, bad)
    try:
        Store(STORE, Native(bp), 2**30)
    except Error as exc:
        assert exc.code == "incompatible"
        results["different_codec"] = exc.code
    else:
        raise AssertionError("incompatible state accepted")
    record(E / "anchors.json", anchors)
    # For both parities freeze exact new token counts from the live template.
    tasks = []
    for label, case in anchors.items():
        for n in (32, 33, 129):
            make = lambda k: "確認番号を数字だけで答えてください。" + (" x" * k)
            msgs, ids = fit(native, case["messages_after"], len(case["ids"]) + n, make)
            assert ids[: len(case["ids"])] == case["ids"]
            ident = f"{label}-{n}"
            infile("question-" + ident, msgs)
            tasks.append(
                {
                    "name": ident,
                    "anchor": label,
                    "new": n,
                    "ids": ids,
                    "max_tokens": 4 if n == 33 else 1,
                }
            )
    record(E / "numeric-inputs.json", tasks)
    # Calibrate ordinary turns with identical inputs and images, no observer. Each pair reverses order.
    task = tasks[1]
    label = task["anchor"]
    case = anchors[label]
    snap = case["snapshot"]
    cost = []
    msgs = load(E / ("question-" + task["name"] + ".json"))
    for i, kind in enumerate(("native", "managed", "managed", "native")):
        if kind == "native":
            native.restore(snap["image_sha256"], db_branch(label)["tokens"], snap["image_bytes"])
            start = time.monotonic_ns()
            ids = native.render(msgs)
            v = manual_stream(native, ids, task["max_tokens"])
            row = {
                "path": kind,
                "first_ms": (v["first_ns"] - start) / 1e6,
                "total_ms": (v["finish_ns"] - start) / 1e6,
                "text": v["content"],
                "cache": v["timings"]["cache_n"],
                "new": v["timings"]["prompt_n"],
            }
        else:
            with manager(STORE, binding, run / f"cost-manager-{i}.log"):
                key = "cost-" + str(i)
                ctl(run, "fork-" + key, "fork", key, "--from", label)
                ctl(run, "checkout-" + key, "checkout", key)
                before = rpc(STORE, {"op": "status"})
                v = ctl(
                    run,
                    "timed-" + key,
                    "turn",
                    "--messages",
                    str(E / ("question-" + task["name"] + ".json")),
                    "--max-tokens",
                    str(task["max_tokens"]),
                )
                after = rpc(STORE, {"op": "status"})
                assert (
                    before["image_bytes"] == after["image_bytes"]
                    and before["resident_images"] == after["resident_images"]
                )
                row = {
                    "path": kind,
                    "first_ms": v["client_first_ms"],
                    "total_ms": v["client_total_ms"],
                    "text": v["content"],
                    "cache": v["timings"]["cache_n"],
                    "new": v["timings"]["prompt_n"],
                    "ordinary_turn_image_bytes_added": 0,
                }
        cost.append(row)
    assert all(v["cache"] == len(case["ids"]) and v["new"] == task["new"] for v in cost)
    results["ordinary_turn"] = cost
    with manager(STORE, binding, run / "timed-checkout.log", R / "fault_manager.py"):
        ctl(run, "setup-switch", "checkout", "B")
        ctl(run, "dirty-B", "turn", "--text", "保管箱番号は変わりません。OK。", "--max-tokens", "2")
        t = time.monotonic_ns()
        sw = ctl(run, "timed-branch-return", "checkout", "A")
        vv = ctl(run, "first-after-return", "turn", "--text", QUERY, "--max-tokens", "8")
        results["dirty_switch_and_first"] = {
            "checkout_ms": sw["client_total_ms"],
            "return_to_first_ms": (vv["client_first_ns"] - t) / 1e6,
            "turn_first_ms": vv["client_first_ms"],
            "replayed": sw["replayed"],
            "text": vv["content"],
        }
        ctl(run, "mark-pre-crash", "mark", "pre-crash")
        pre = rpc(STORE, {"op": "status"})["ticket"]
        record(plan, {"mode": "crash-after-confirmed-restore", "count": 1})
        ctl(run, "interrupted-checkout", "checkout", "main", expect="manager_disconnected")
    with manager(STORE, binding, run / "fenced-manager.log"):
        held = ctl(run, "interrupted-reopen", "status")
        assert held["mode"] == "uncertain" and held["ticket"] == pre
        ctl(run, "fence-ambiguous-backend", "recover", expect="native_outcome_unknown")
    results["crash_retained_ticket"] = pre
    record(E / "deep.json", results)


def numeric(run, binding):
    tasks = load(E / "numeric-inputs.json")
    recovered = load(E / "deep.json")["crash_retained_ticket"]
    with manager(STORE, binding, run / "manager.log"):
        st = ctl(run, "reopen-after-interruption", "status")
        assert st["mode"] == "ready" and st["ticket"]["branch"] == recovered["branch"]
        assert st["ticket"]["generation"] == recovered["generation"]
        for t in tasks:
            lab = t["name"]
            name = "score-" + lab
            ctl(run, name + "-fork", "fork", name, "--from", t["anchor"])
            v = ctl(run, name + "-checkout", "checkout", name)
            assert v["exact_head"] and v["replayed"] == 0
            phase(run, name)
            try:
                v = ctl(
                    run,
                    name,
                    "turn",
                    "--messages",
                    str(E / ("question-" + lab + ".json")),
                    "--max-tokens",
                    str(t["max_tokens"]),
                )
            finally:
                phase(run, "idle")
            assert v["timings"]["prompt_n"] == t["new"]
        # Explicitly force an LRU eviction on synthetic state; pinned parent stays available.
        ctl(run, "budget-preset-active", "checkout", "main")
        initial = rpc(STORE, {"op": "status"})
        parent = load(E / "basic.json")["P"]
        v = ctl(run, "limited-budget", "budget", str(parent["image_bytes"] + 1024))
        assert v["evicted_images"]
        checked = ctl(run, "budget-check", "status")
        assert checked["image_bytes"] <= checked["budget_bytes"]
        assert (
            ctl(run, "pinned-survives", "inspect", "snapshot", "parent")["image_status"]
            == "available"
        )
        ctl(run, "restore-image-budget", "budget", str(initial["budget_bytes"]))
        v = ctl(run, "evicted-replay", "checkout", "odd")
        assert v["replayed"] > 0
        ctl(run, "roundtrip-evicted", "mark", "rebuilt-after-budget")
        record(
            E / "budget.json",
            {
                "before": initial["image_bytes"],
                "after": checked["image_bytes"],
                "cap": checked["budget_bytes"],
                "pinned_bytes": parent["image_bytes"],
                "recovery": v,
            },
        )


def stock(run, binding):
    native = Native(binding)
    tasks = load(E / "numeric-inputs.json")
    anchors = load(E / "anchors.json")
    out = []
    for label, anchor in anchors.items():
        snap = anchor["snapshot"]
        dst = native.images / (snap["image_sha256"] + ".slot")
        shutil.copyfile(E / "references" / (label + ".slot"), dst)
        dst.chmod(0o444)
        assert digest(dst) == snap["image_sha256"]
    for t in tasks:
        label = t["anchor"]
        anchor = anchors[label]
        snap = anchor["snapshot"]
        native.restore(snap["image_sha256"], pack(anchor["ids"]), snap["image_bytes"])
        name = "score-" + t["name"]
        ids = native.render(load(E / ("question-" + t["name"] + ".json")))
        assert ids == t["ids"]
        phase(run, name)
        try:
            v = manual_stream(native, ids, t["max_tokens"])
        finally:
            phase(run, "idle")
        assert v["timings"]["prompt_n"] == t["new"] and v["timings"]["cache_n"] == len(
            anchor["ids"]
        )
        out.append({"name": name, **v})
    record(E / "manual-stock.json", out)
