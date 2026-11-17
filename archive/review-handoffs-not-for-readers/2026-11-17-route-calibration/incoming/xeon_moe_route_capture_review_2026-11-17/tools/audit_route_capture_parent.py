#!/usr/bin/env python3
"""Read-only replay of archived route capture evidence; never runs an engine.

Use a worktree pinned to caf35bcdd27c0240572e05a60633f07fca272844.
This checks saved bytes, ledgers, repetition, and source-implied export work.
It is not a new model test, timer, GPU admission test, or bank optimizer.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

PIN = "caf35bcdd27c0240572e05a60633f07fca272844"
REL = Path("archive/review-handoffs-not-for-readers/2026-11-17-route-capture")
RUN_NAMES = ("20261117T000246Z-d4e0df3a", "20261117T001035Z-68c12e67")


def read_json(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prefix_report(run, n):
    value = deepcopy(run)
    value["passed"] = False
    value.pop("session_closed_before_owner_destruction")
    value["call_results"] = value["call_results"][:n]
    trace = value["trace"]
    trace["records"] = trace["records"][:37*n]
    trace["attempts"] = trace["attempts"][:n]
    trace["calls"] = trace["calls"][:n]
    trace["stats"].update(records=37*n, attempts=n, calls=n,
                          complete_decode_attempts=n, enabled=True)
    return value


def inspect_run(path):
    run = read_json(path / "RUN.json")
    n = run["planned_capture_calls"]
    assert n in (4, 68)
    assert run["passed"] and run["model_routes"]
    assert run["session_closed_before_owner_destruction"]
    assert run["prefix_tokens"] == 64
    assert len(run["tokens"]) == 64+n
    trace = run["trace"]
    expected_stats = dict(records=37*n, attempts=n, calls=n, dropped_events=0,
                          complete_decode_attempts=n, flags=0, enabled=False,
                          observing=True)
    assert trace["stats"] == expected_stats
    assert len(trace["records"]) == 37*n
    assert len(trace["attempts"]) == len(trace["calls"]) == len(run["call_results"]) == n
    sets_by_call = []
    for t, (attempt, call, result) in enumerate(zip(trace["attempts"], trace["calls"], run["call_results"])):
        label = dict(request=0, sequence=0, decode_call=t, phase=2, logical_input_token=t)
        assert attempt["label"] == call["label"] == label
        for k, v in dict(id=t+1, generation=1, first_record=37*t, records=37,
                         layers=(1 << 37)-1, position=64+t, context=1,
                         ubatch=0, n_tokens=1, flags=0, graph_status=0,
                         observing=True, scope_t1=True, graph_returned=True,
                         caller_completed=True).items():
            assert attempt[k] == v, (t, "attempt", k)
        for k, v in dict(first_attempt=t, attempts=1, flags=0, public_or_driver_status=0,
                         synchronized=True, completed=True).items():
            assert call[k] == v, (t, "call", k)
        assert result == dict(logical_input_token=t, position=64+t, label_accepted=True,
                              decode_returned=True, decode_status=0, output_called=True,
                              output_valid=True, synchronized=True, drain_called=False,
                              finalize_called=True, finalize_ok=True, exception_stage=0, ok=True)
        row = []
        for layer, record in enumerate(trace["records"][37*t:37*(t+1)]):
            expected = dict(ordinal=37*t+layer, attempt_id=t+1, request=0, sequence=0,
                            decode_call=t, ubatch=0, logical_input_token=t, position=64+t,
                            generation=1, context=1, layer=layer, phase=2, projection=1,
                            n_tokens=1, flags=0)
            assert {k: v for k, v in record.items() if k != "ids"} == expected
            ids = record["ids"]
            assert len(ids) == len(set(ids)) == 6
            assert all(isinstance(x, int) and 0 <= x < 384 for x in ids)
            row.append(set(ids))
        sets_by_call.append(row)
    assert dump_bytes(run) == (path/"RUN.json").read_bytes()
    assert dump_bytes(prefix_report(run, 4)) == (path/"SMOKE4.json").read_bytes()
    per_call_json_bytes = [len(dump_bytes(prefix_report(run, i))) for i in range(1, n+1)]
    tokens = run["tokens"][64:]
    periods = [p for p in range(1, n) if all(tokens[i] == tokens[i-p] for i in range(p, n))]
    lags = {}
    for lag in (1, 22, 44):
        if lag >= n:
            continue
        values = [len(sets_by_call[t][l] & sets_by_call[t-lag][l])
                  for t in range(lag, n) for l in range(37)]
        hist = Counter(values)
        lags[str(lag)] = dict(pairs=len(values), histogram=[hist[h] for h in range(7)],
                              mean_common_experts=sum(values)/len(values))
    per_layer_unique = [len(set().union(*(x[l] for x in sets_by_call))) for l in range(37)]
    log = (path/"server.stderr.log").read_bytes().decode("utf-8", errors="replace")
    def printed_int(field):
        m = re.search(r"\b"+field+r"\s*=\s*(\d+)", log)
        assert m, field
        return int(m.group(1))
    summary = dict(run=path.name, calls=n, records=37*n, expert_id_slots=37*n*6,
                   ledger_validated=True, minimal_finite_word_period=min(periods) if periods else None,
                   continuation_unique_token_ids=len(set(tokens)), lag_overlap=lags,
                   unique_experts_per_layer=per_layer_unique,
                   raw_run_bytes=(path/"RUN.json").stat().st_size,
                   raw_run_sha256=sha((path/"RUN.json").read_bytes()),
                   json_format_and_smoke_reconstructed_byte_exact=True,
                   per_call_serialized_write_request_bytes=sum(per_call_json_bytes),
                   old_per_call_record_visits=37*n*(n+1)//2,
                   old_snapshot_record_visits_including_final=37*n*(n+1)//2+37*n,
                   proposed_snapshot_record_visits_4_and_final=37*(4+n),
                   n_vocab=printed_int("n_vocab"), n_outputs_max=printed_int("n_outputs_max"),
                   n_batch=printed_int("n_batch"), n_ubatch=printed_int("n_ubatch"))
    return run, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.repo / REL
    manifest = read_json(base/"FILES_SHA256.json")
    for name, entry in manifest.items():
        data = (base/name).read_bytes()
        assert len(data) == entry["bytes"], name
        assert sha(data) == entry["sha256"], name
    small, s = inspect_run(base/"runs"/RUN_NAMES[0])
    large, l = inspect_run(base/"runs"/RUN_NAMES[1])
    for key, length in (("records", 148), ("attempts", 4), ("calls", 4)):
        assert small["trace"][key] == large["trace"][key][:length]
    assert small["call_results"] == large["call_results"][:4]
    assert small["tokens"] == large["tokens"][:68]
    restored = read_json(base/"RESTORED_RUNTIME.json")
    result = dict(schema=1, pinned_commit=PIN, manifest_entries_checked=len(manifest),
                  first_four_calls_all_record_fields_equal=True, runs=[s, l],
                  restored_runtime_receipt=restored,
                  limitations=["Saved-data/source replay only; no new engine, model, GPU or API execution.",
                               "No timing benchmark, no actual logits values, no language comparison.",
                               "A finite-word period is a property of this fixed token fixture only.",
                               "Export bytes are source-implied serialization/write requests, not physical I/O or milliseconds.",
                               "The 4-call run overlaps the 68-call prefix; they are not 72 independent samples."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.out), manifest_entries_checked=len(manifest),
                          first_four_calls_equal=True, runs=[s,l]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
