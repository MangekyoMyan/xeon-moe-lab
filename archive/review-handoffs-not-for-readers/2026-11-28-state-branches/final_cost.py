"""Uninstrumented final CLI calibration with the same starting images on both paths."""

import time
from lab_trial import R, E, Native, ctl, log, manager, rpc, record, load, QUERY, PARENT, register
from more_trials import fit, manual_stream
from native import image_tokens


def final_cost(run, binding, store):
    n = Native(binding)

    def cli(step, *args, **kw):
        return ctl(run, step, *args, store=store, **kw)

    with manager(store, binding, run / "prepare.log"):
        case = load(E / "anchors.json")["odd"]
        seeded = cli("seed", "turn", "--messages", str(E / "anchor-odd.json"), "--max-tokens", "1")
        P = cli("save-parent", "mark", "P", "--pin")["snapshot"]
        pmsg = case["messages"] + [{"role": "assistant", "content": seeded["content"]}]
        cli("fork-target", "fork", "T", "--from", "P")
        cli("enter-target", "checkout", "T")
        answer = cli("register-target", "turn", "--text", register("731"), "--max-tokens", "1")
        T = cli("save-target", "mark", "T0", "--pin")["snapshot"]
        tmsg = pmsg + [
            {"role": "user", "content": register("731")},
            {"role": "assistant", "content": answer["content"]},
        ]
        cli("enter-parent", "checkout", "main")

    def restore(point):
        blob = n.images / (point["image_sha256"] + ".slot")
        n.restore(point["image_sha256"], image_tokens(blob, n.limit), point["image_bytes"])

    numeric = next(v for v in load(E / "numeric-inputs.json") if v["name"] == "odd-33")
    msgs, _ = fit(n, pmsg, 288, lambda k: "確認番号を数字だけで答えてください。" + (" x" * k))
    record(E / "final-cost-input.json", msgs)
    ordinary = []
    for i, which in enumerate(
        ("native", "managed", "managed", "native", "managed", "native", "native", "managed")
    ):
        if which == "native":
            restore(P)
            start = time.monotonic_ns()
            ids = n.render(msgs)
            out = manual_stream(n, ids, numeric["max_tokens"])
            entry = {
                "path": which,
                "first_ms": (out["first_ns"] - start) / 1e6,
                "total_ms": (out["finish_ns"] - start) / 1e6,
                "content": out["content"],
                "cache": out["timings"]["cache_n"],
                "new": out["timings"]["prompt_n"],
            }
        else:
            with manager(store, binding, run / f"turn-{i}.log"):
                name = "round" + str(i)
                cli("fork-" + name, "fork", name, "--from", "P")
                cli("checkout-" + name, "checkout", name)
                bef = rpc(store, {"op": "status"})
                out = cli(
                    "ordinary-" + name,
                    "turn",
                    "--messages",
                    str(E / "final-cost-input.json"),
                    "--max-tokens",
                    str(numeric["max_tokens"]),
                )
                aft = rpc(store, {"op": "status"})
                assert (
                    bef["image_bytes"] == aft["image_bytes"]
                    and bef["resident_images"] == aft["resident_images"]
                )
                entry = {
                    "path": which,
                    "first_ms": out["client_first_ms"],
                    "total_ms": out["client_total_ms"],
                    "content": out["content"],
                    "cache": out["timings"]["cache_n"],
                    "new": out["timings"]["prompt_n"],
                    "image_bytes_added": 0,
                }
        ordinary.append(entry)
    assert all((x["cache"], x["new"]) == (255, 33) for x in ordinary)
    assert len({x["content"] for x in ordinary}) == 1
    # Both paths first make the same unsaved departure and restore the same target.
    cycles = []
    for i, which in enumerate(("managed", "native", "native", "managed")):
        dirty = "この枝で扱う仮の控え番号は 842 です。OK とだけ答えてください。"
        target_msgs = tmsg + [{"role": "user", "content": QUERY}]
        if which == "native":
            restore(P)
            ids = n.render(pmsg + [{"role": "user", "content": dirty}])
            manual_stream(n, ids, 1)
            start = time.monotonic_ns()
            path = n.images / "manual-working.slot"
            try:
                receipt = n.call("POST", "/slots/0?action=save", {"filename": path.name})
                restore(T)
                installed = time.monotonic_ns()
                out = manual_stream(n, n.render(target_msgs), 8)
                resident = sum(f.stat().st_size for f in n.images.glob("*.slot"))
            finally:
                path.unlink(missing_ok=True)
            cycles.append(
                {
                    "path": which,
                    "switch_ms": (installed - start) / 1e6,
                    "return_to_first_ms": (out["first_ns"] - start) / 1e6,
                    "content": out["content"],
                    "saved_tokens": receipt["n_saved"],
                    "saved_bytes": receipt["n_written"],
                    "resident_at_measurement": resident,
                    "cached": out["timings"]["cache_n"],
                    "new": out["timings"]["prompt_n"],
                }
            )
        else:
            with manager(store, binding, run / f"switch-{i}.log"):
                dep = f"depart{i}"
                dst = f"dest{i}"
                cli("fork-" + dep, "fork", dep, "--from", "P")
                cli("fork-" + dst, "fork", dst, "--from", "T0")
                cli("activate-" + dep, "checkout", dep)
                cli("dirty-" + dep, "turn", "--text", dirty, "--max-tokens", "1")
                start = time.monotonic_ns()
                ch = rpc(
                    store,
                    {
                        "op": "checkout",
                        "name": dst,
                        "ticket": rpc(store, {"op": "ticket"})["ticket"],
                    },
                )
                installed = time.monotonic_ns()
                seen = []
                out = rpc(
                    store,
                    {
                        "op": "turn",
                        "text": QUERY,
                        "max_tokens": 8,
                        "ticket": rpc(store, {"op": "ticket"})["ticket"],
                    },
                    lambda _: seen.append(time.monotonic_ns()),
                )
                outgoing = rpc(
                    store, {"op": "inspect", "kind": "snapshot", "name": ch["outgoing_snapshot"]}
                )
                st = rpc(store, {"op": "status"})
                assert ch["exact_head"] and ch["replayed"] == 0
                cycles.append(
                    {
                        "path": which,
                        "switch_ms": (installed - start) / 1e6,
                        "return_to_first_ms": (seen[0] - start) / 1e6,
                        "content": out["content"],
                        "saved_tokens": outgoing["processed"],
                        "saved_bytes": outgoing["image_bytes"],
                        "resident_at_measurement": st["image_bytes"],
                        "cached": out["timings"]["cache_n"],
                        "new": out["timings"]["prompt_n"],
                    }
                )
    assert all(x["resident_at_measurement"] <= 1024**3 for x in cycles)
    assert (
        len(
            {
                (x["saved_tokens"], x["saved_bytes"], x["cached"], x["new"], x["content"])
                for x in cycles
            }
        )
        == 1
    )
    # A text containing a UTF-8 sequence split into native tokens can be displayed in fewer events.
    unicode_rows = []
    with manager(store, binding, run / "unicode.log"):
        for i, text in enumerate(("🪿", "𩸽", "🦦")):
            v = cli(
                "unicode-" + str(i),
                "turn",
                "--text",
                "次の一文字をそのまま一回だけ表示してください。 " + text,
                "--max-tokens",
                "12",
            )
            saved = cli("unicode-mark-" + str(i), "mark", "unicode" + str(i))["snapshot"]
            assert saved["processed"] == v["processed"]
            unicode_rows.append(
                {
                    "requested": text,
                    "answer": v["content"],
                    "n_sampled": v["sampled_count"],
                    "ids_in_partial_events": v["partial_stream_token_count"],
                    "pending": v["pending_sampled_ids"],
                    "verified_saved_tokens": saved["processed"],
                }
            )
            if (
                v["partial_stream_token_count"] < v["sampled_count"]
                and v["content"].strip() == text
            ):
                break
    record(
        E / "final-cost.json",
        {
            "ordinary": ordinary,
            "dirty_checkout": cycles,
            "unicode": unicode_rows,
            "image_budget": 1024**3,
        },
    )
