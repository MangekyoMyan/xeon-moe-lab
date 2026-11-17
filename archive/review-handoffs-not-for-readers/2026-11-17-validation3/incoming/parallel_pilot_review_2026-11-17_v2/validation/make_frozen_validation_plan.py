#!/usr/bin/env python3
"""Extract a declarative validation plan from saved files. No model/tokenizer/API execution."""
import argparse
import ast
import hashlib
import json
import struct
from pathlib import Path

PIN = "d4d50b651ff7754ee2189228762468fc02971bfa"
TOKEN_SHA = "44a9f673a7928b30f29a11fd741c96fa5b2faf526b89a82bc448aa13ff32596f"
CORPUS_SHA = "fa1f41571af7c6d8272bb639877ea0cc32b4c09747939d5579a22d2998c11f5a"
BANK_SHA = "4b5a8063ae1e61c55d30f957798a7cc44d7f462cf22767d2596105ed7322c596"
SELECTED = ["text_plan_constraints_ja", "text_plan_constraints_en", "code_bounded_queue_python"]
ARCH = Path("archive/review-handoffs-not-for-readers")
W = ARCH / "2026-11-17-gpu-workspace"
N = ARCH / "2026-11-17-parallel-pilot"
CORPUS_DIR = W / "incoming/xeon_moe_route_calibration_review_2026-11-17/pilot"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def i32sha(ids):
    return sha(struct.pack("<" + "i" * len(ids), *ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def bytes_from(rel, expected=None):
        data = (repo / rel).read_bytes()
        h = sha(data)
        if expected:
            assert h == expected, (str(rel), "source SHA mismatch", h)
        sources[str(rel)] = {"sha256": h, "bytes": len(data)}
        return data

    def read(rel, expected=None):
        return json.loads(bytes_from(rel, expected))

    tokens = read(W / "PILOT_TOKENS.json", TOKEN_SHA)
    corpus = read(CORPUS_DIR / "PILOT_CASES_JA_EN_CODE.json", CORPUS_SHA)
    stage = read(CORPUS_DIR / "PILOT_STAGE_MANIFEST.json")
    design = read(W / "pilot-capture/PILOT_DESIGN.json")
    bank_bytes = bytes_from(N / "integrated/capacity.plan", BANK_SHA)
    assert bytes_from(N / "replay/capacity.plan", BANK_SHA) == bank_bytes
    bank_design = read(N / "replay/BANK_DESIGN.json")
    integration_design = read(N / "integrated/DESIGN.json")
    bytes_from(W / "RESULTS_JA.txt")
    bytes_from(N / "RESULTS_JA.txt")
    for rel in [N / "integrated/driver.cpp", N / "analyze_integrated.py", N / "integrated/run_integrated.py"]:
        bytes_from(rel)

    assert tokens["passed"] and tokens["corpus_sha256"] == CORPUS_SHA
    assert tokens["model_inference_requested"] is False
    assert tokens["route_collected"] is False and tokens["bank_fitted"] is False
    assert tokens["split_policy"] == "Original family splits retained; no outcome-based adjustment"
    assert stage["corpus_sha256"] == CORPUS_SHA
    assert stage["cases_by_split"] == {"train": 10, "validation": 5, "holdout": 5}
    assert design["stage"] == "train_first5 only"
    assert design["all_token_manifest_sha256"] == TOKEN_SHA
    assert "validation5" in design["withheld_routes"] and "holdout5" in design["withheld_routes"]
    assert integration_design["corpus"] == "only JA text_cache_budget_ja, prefix272+reference64"
    validation_ids = next(x["cases"] for x in stage["stages"] if x["stage"] == "validation5")
    holdout_ids = next(x["cases"] for x in stage["stages"] if x["stage"] == "holdout5")
    assert set(SELECTED) <= set(validation_ids)
    assert not set(SELECTED) & set(holdout_ids)
    assert not set(SELECTED) & set(design["case_ids"])

    # Inspect shipped execution reports, including failed capture attempts. Their case lists
    # contain only wiring/capacity and the five training cases, never these validation cases.
    run_receipts = []
    for folder in [W, N]:
        for path in sorted((repo / folder).glob("**/runs/*/RUN.json")):
            rel = path.relative_to(repo)
            doc = read(rel)
            ids = [c.get("case_id", c.get("name")) for c in doc.get("cases", [])]
            overlap = sorted(set(ids) & set(SELECTED))
            assert not overlap, (str(rel), "selected validation case already in shipped run", overlap)
            run_receipts.append({"path": str(rel), "sha256": sources[str(rel)]["sha256"],
                                 "passed": doc.get("passed"), "declared_case_ids": ids,
                                 "selected_validation_case_overlap": overlap})
    assert len(run_receipts) == 10

    bank_text = bank_bytes.decode()
    bank_lines = bank_text.splitlines()
    hdr = bank_lines[0].split()
    assert hdr == ["XEON_HOT_FFN_PLAN_V1", "40", "384", "5120", "2304", "35", "475453440"]
    parsed_rows = []
    for line in bank_lines[1:]:
        fields = line.split()
        layer, count = int(fields[0]), int(fields[4])
        ids = list(map(int, fields[5:]))
        assert count == len(ids) == len(set(ids))
        parsed_rows.append({"layer": layer, "types_gate_up_down": fields[1:4], "expert_ids": ids})
    assert len(parsed_rows) == 40 and sum(len(x["expert_ids"]) for x in parsed_rows) == 35
    assert sum(bool(x["expert_ids"]) for x in parsed_rows) == 23
    assert all(not row["expert_ids"] for row in parsed_rows if row["layer"] >= 37)
    assert bank_design["plan_sha256"] == BANK_SHA and bank_design["native_bytes"] == 475453440

    token_by_id = {c["case_id"]: c for c in tokens["cases"]}
    corpus_by_id = {v["case_id"]: (f, v) for f in corpus["families"] for v in f["variants"]}
    assert len(token_by_id) == len(corpus_by_id) == 20
    cases, checks = [], []
    for cid in SELECTED:
        frozen = token_by_id[cid]
        family, variant = corpus_by_id[cid]
        assert frozen["split"] == family["split"] == "validation"
        assert family["split_frozen_before_routes"]
        assert frozen["family_id"] == family["family_id"]
        assert frozen["language_or_content"] == variant["language_or_content"]
        for key in ["prompt", "reference"]:
            text = variant[key + "_utf8"]
            assert sha(text.encode()) == variant[key + "_sha256"] == frozen[key + "_sha256"]
            assert len(text.encode()) == variant[key + "_utf8_bytes"]
            assert len(text) == variant[key + "_codepoints"]
        prefix, continuation = frozen["prefix_ids"], frozen["full_reference_continuation_ids"]
        n = frozen["prefix_count"]
        assert n == len(prefix) == frozen["first_T1_position"]
        assert len(continuation) == frozen["reference_count"] >= 65
        assert frozen["strict_prefix_verified"] is True
        assert continuation[:65] == frozen["continuation_first65_ids"] == frozen["next_targets_prefix_and64T1"]
        assert continuation[:64] == frozen["used_T1_ids"]
        assert i32sha(prefix) == frozen["prefix_ids_i32_sha256"]
        assert i32sha(continuation[:64]) == frozen["used_T1_ids_i32_sha256"]
        assert sha(frozen["rendered_prefix"].encode()) == frozen["rendered_prefix_sha256"]
        joined = frozen["rendered_prefix"] + variant["reference_utf8"]
        assert sha(joined.encode()) == frozen["joined_text_sha256"]
        assert frozen["rendered_prefix"].endswith(frozen["assistant_boundary_suffix"])
        assert frozen["decoded_first64"].encode().hex() == frozen["decoded_first64_bytes_hex"]
        all_ids = prefix + continuation
        assert all(type(x) is int and 0 <= x < 129280 for x in all_ids)
        if cid == "code_bounded_queue_python":
            assert sha(variant["reference_code_utf8"].encode()) == variant["reference_code_sha256"]
            ast.parse(variant["reference_code_utf8"], filename="frozen_reference_only.py")
        if family["family_id"] == "text_plan_constraints":
            assert family["common_semantic_contract"]["earliest_finish"] == "10:45"
            assert sum(family["common_semantic_contract"]["tasks"][k]["minutes"]
                       for k in ["S", "I", "E", "F"]) == 105
        cases.append({
            "case_id": cid, "family_id": family["family_id"], "split": "validation",
            "family_metadata_exact": {k: v for k, v in family.items() if k != "variants"},
            "original_corpus_variant_exact": variant,
            "frozen_token_manifest_case_exact": frozen,
            "execution_input": {
                "prefix_count": n, "prefix_ids": prefix,
                "prompt_utf8": variant["prompt_utf8"], "reference_utf8": variant["reference_utf8"],
                "rendered_prefix_utf8": frozen["rendered_prefix"], "joined_text_utf8": joined,
                "joined_expected_ids": all_ids, "joined_expected_ids_i32_sha256": i32sha(all_ids),
                "teacher_forced_input64": continuation[:64],
                "teacher_forced_next_targets65": continuation[:65],
                "input_absolute_positions64": list(range(n, n + 64)),
                "row_target_mapping": "row0 after prefill predicts continuation[0]; row(i+1) after input continuation[i] predicts continuation[i+1], i=0..63",
                "timing_calls": 64, "quality_rows": 65, "vocab_size": 129280,
                "quality_row_bytes": 129280 * 4, "quality_file_bytes": 65 * 129280 * 4,
            },
            "free_answer_rubric_id": "schedule_constraints" if family["family_id"] == "text_plan_constraints" else "ring_queue_python",
        })
        checks.append({"case_id": cid, "prefix_tokens": n, "reference_tokens": len(continuation),
                       "input_count": 64, "next_target_count": 65,
                       "T1_positions_inclusive": [n, n + 63], "split": "validation"})

    rubrics = {
        "grading_policy": "Record pass/fail/incomplete per criterion plus the full output. Do not replace task correctness by exact reference wording or top1 agreement. No general-quality or language-wide claim from these three variants/two families.",
        "schedule_constraints": {
            "expected_earliest_finish": "10:45", "critical_path": ["S", "I", "E", "F"],
            "critical_path_minutes": 105,
            "canonical_feasible_schedule": {"S": ["09:00", "09:20"], "I": ["09:20", "09:50"],
                                             "E": ["09:50", "10:30"], "R": ["09:50", "10:15"],
                                             "F": ["10:30", "10:45"]},
            "alternative_optima": "Accept any schedule satisfying all stated constraints and finishing at10:45. R need not start at09:50; a25-minute interval starting between09:50 and10:05 and ending by10:30 is also optimal.",
            "criteria": [
                "All five tasks have explicit start/end times, correct durations, assigned people and precedence.",
                "Every task lies within09:00..11:00, no person is double-booked; tasks are uninterrupted and durations/people are unchanged.",
                "E and R may overlap because assigned people are disjoint; no invented room/equipment constraint.",
                "F waits for both E and R, uses all three, and ends at10:45.",
                "A lower-bound proof uses S->I->E->F=20+30+40+15=105minutes; feasible schedule attains it.",
            ],
            "languages": "Use the same semantic rubric for Japanese and English. Treat the pair as one family, not two independent tasks proving a language effect.",
        },
        "ring_queue_python": {
            "interface": ["RingQueue(capacity)", "size()", "try_push(value)", "try_pop()"],
            "preconditions": "Integer inputs excluding bool and signed64 representation are already guaranteed. Require capacity-range checks, not unrelated type checking.",
            "criteria": [
                "Accept capacities1 and1024; reject0, negative values and1025 or larger, including signed64 endpoints outside the interval.",
                "Fixed-capacity ring buffer with oldest-element index and count; no shifting/pop(0) replacement for the specified ring representation.",
                "size() equals current element count, 0<=size<=capacity through all transitions.",
                "Successful pushes/pops preserve FIFO including zero, negative and repeated values, signed64 minimum and maximum.",
                "Push when full returns False and leaves observable state unchanged; empty pop returns explicit empty Optional and leaves state unchanged.",
                "Conventional Optional[int] uses None for empty; an explicit documented optional representation is acceptable only if it unambiguously distinguishes stored0 from empty.",
                "Use an oldest-element index and count to wrap within fixed capacity while preserving FIFO. The exact index formula and physical traversal direction are not prescribed; failed operations leave the logical queue unchanged.",
                "Given capacity2 example: pushes4 and9 add the values; push12 returns False without state change; pop returns4; push(-3) adds the value; pops return9 and-3, then explicit empty. The prompt does not prescribe a successful push return value; assess its size/FIFO effects rather than requiring True.",
                "Capacity1 alternation and repeated wraparound work; standard library only; interface independent of printing/file I/O.",
                "The answer provides complete parseable Python code and explains wraparound and failure-state preservation.",
            ],
            "future_behavior_vectors": [
                {"capacity": 2, "operations": [["push", 4], ["push", 9], ["push", 12], ["pop"], ["push", -3], ["pop"], ["pop"], ["pop"]],
                 "expected_semantic_results": [{"push_succeeds": True}, {"push_succeeds": True}, False, 4, {"push_succeeds": True}, 9, -3, {"empty_optional": True}]},
                {"capacity": 1, "operations": [["pop"], ["push", 0], ["push", 99], ["size"], ["pop"], ["pop"], ["push", -9223372036854775808], ["pop"], ["push", 9223372036854775807], ["pop"]],
                 "expected_semantic_results": [{"empty_optional": True}, {"push_succeeds": True}, False, 1, 0, {"empty_optional": True}, {"push_succeeds": True}, -9223372036854775808, {"push_succeeds": True}, 9223372036854775807]},
                {"capacity": 3, "operations": [["push", -2], ["push", -2], ["push", 0], ["pop"], ["push", -2], ["pop"], ["pop"], ["pop"], ["size"]],
                 "expected_semantic_results": [{"push_succeeds": True}, {"push_succeeds": True}, {"push_succeeds": True}, -2, {"push_succeeds": True}, -2, 0, -2, 0]},
                {"capacity": 1024, "procedure": "Push values i-512 for i=0..1023; verify full push fails and size1024; pop all in order; verify empty. Then interleave pushes/pops for more than2*capacity operations to cover wraparound."},
            ],
            "semantic_marker_rules": "push_succeeds means the item is appended, size rises by one and later pops preserve FIFO; its return value is unspecified by the prompt. Boolean False marks the required full-push result. empty_optional means any documented explicit empty result distinguishable from stored zero. Do not compare a dict marker directly with the method return.",
            "execution_status": "These are future grading specifications only; generated or supplied code is not executed by this extraction script.",
        },
    }

    plan = {
        "schema": 1, "kind": "declarative_frozen_validation_plan_not_an_engine_runner",
        "review_revision": 2,
        "supersedes_plan_sha256": "967c653002983016c3e50872a308ff3fae367e5fb84edd7aaee6df6195408dbc",
        "status": "proposal_not_executed", "source_repository_pin": PIN,
        "purpose_ja": "固定35expert案について、候補選定に未使用の3例でF/Pの時間と実回答を確認し、S/Pの同時実行差を別の数値blockで検査する。全mode×全caseの性能gridは作らない。",
        "selected_case_ids_in_fixed_order": SELECTED,
        "case_count": 3, "independent_family_count": 2,
        "no_model_or_tokenizer_execution_during_preparation": True,
        "source_manifest": sources,
        "original_corpus_global_metadata_exact": {k: v for k, v in corpus.items() if k != "families"},
        "frozen_token_global_metadata_exact": {k: v for k, v in tokens.items() if k != "cases"},
        "metadata_scope_note": "original_corpus_variant_exact predates token freeze and intentionally retains tokenization_performed=false/null IDs; frozen_token_manifest_case_exact is the later saved token record. These fields are preserved rather than rewritten.",
        "unused_validation_evidence": {
            "claim": "These cases are unused for model inference/routing/selection in the shipped W and N records at the pinned commit; all20 were previously rendered/tokenized only.",
            "family_split_frozen_before_routes": True,
            "token_freeze_flags": {k: tokens[k] for k in ["model_inference_requested", "route_collected", "bank_fitted", "split_policy"]},
            "train_design_exact": design, "current_integrated_design_exact": integration_design,
            "reviewed_execution_reports": run_receipts,
            "limitation": "A repository snapshot cannot prove absence of unexported or later real-machine runs. Before actual use, record whether these exact cases have since been run; do not relabel them fresh if used.",
        },
        "bank": {"name": "original35", "native_bytes": 475453440, "expert_count": 35,
                 "active_cpu_layers": 23, "plan_sha256": BANK_SHA, "plan_text_exact": bank_text,
                 "parsed_rows": parsed_rows, "reselection_allowed": False,
                 "GPU_full_layers_retained": [37, 38, 39],
                 "validation_route_based_retuning_allowed": False,
                 "holdout": "No holdout prompt, token IDs, inference or route collection is scheduled; do not substitute holdout cases or reselect35 after looking at validation."},
        "cases": cases,
        "fixed_runtime_contract": {
            "F": "Same integrated trial libllama, additional hot path disabled; preserve existing GPU full layers37..39.",
            "S": "Same35 bank and partition asP, serial CPU/GPU work; numerical diagnosis only, no timedS matrix.",
            "P": "Same35 bank, concurrent CPU/GPU work.",
            "same_across_arms": ["model weights", "CPU/CUDA/base libraries", "CPU compact plan", "threads12/24", "NUMA policy", "ctx131072", "f16KV", "n_batch1024", "n_ubatch1024", "output cap1", "one slot", "GPU full layers37..39", "chat template and exact prefix IDs"],
            "preserve_existing_operating_policy": "150W/ECC/fan<=70, RAM reserve64GiB and existing guards; no ubatch sweep or CPUabs candidate.",
            "initialization": "State/bank once per context; switch modes only after completed sync; clearKV and prefill exact same prefix before every independent block/answer.",
            "observation": "route collector closed and callback=null in this plan; no logical/private routing claims until origin separation is implemented. Record execution counters outside timing.",
            "build_provenance": "The case-selection change requires a new reviewed build/hash receipt. This plan does not claim the current driver already runs it or replace prior frozen binary pins.",
        },
        "stages": {
            "quality_first": {
                "modes": ["F", "S", "P"], "purpose": "S/P byte equivalence; F/P full-vector numerical description on identical fixed reference inputs.",
                "blocks_per_mode_per_case": 1, "calls_per_block": 64,
                "full_f32_rows_per_block": 65, "shape": [65, 129280],
                "diagnostic_T1_calls_total": 576,
                "outside_timing": ["finite scans", "F32 copies", "hashes", "exports", "quality analysis"],
                "required_check": "All saved vectors finite; S andP identical byte-for-byte for each case on all65rows. If not, retain raw and stop performance/free-answer progression to investigate.",
                "F_P_metrics": ["per-row and aggregate NMSE", "max absolute difference", "top1 IDs and agreement", "same65 target logprobs/NLL difference using stable full-vocabulary logsumexp"],
                "numerical_policy": "No F/P bit-equality requirement and no invented acceptance threshold. Top1agreement/NMSE are not answer-quality scores.",
            },
            "teacher_forced_timing": {
                "primary_modes": ["F", "P"], "calls_per_block": 64,
                "independent_process_order": [
                    {"process_index": 0, "mode": "F", "case_order": SELECTED},
                    {"process_index": 1, "mode": "P", "case_order": SELECTED},
                    {"process_index": 2, "mode": "P", "case_order": list(reversed(SELECTED))},
                    {"process_index": 3, "mode": "F", "case_order": list(reversed(SELECTED))},
                ],
                "blocks_per_mode_per_case": 2, "timed_T1_calls_total": 768,
                "process_local_preparation": "Each timing process must perform an explicitly recorded untimed T1 preparation with the same input/count policy for F and P. Quality in another process and private-graph initialization are not substitutes for main-graph preparation/first-use logs. Synchronize and clearKV, refill the exact target prefix, then take counter before-snapshots and preserve all64 timed T1 including the first. Freeze the short preparation policy before inspecting gains; do not add a large sweep or discard measured early calls after the fact.",
                "boundary": "llama_decode begin through normal host logits/output-ready synchronization; use start_ns/output_ready_ns integer pairs exactly as the prior pilot.",
                "retain": "Every call including firstT1, all12blocks, actual order and timestamps. Keep prefill/load/warmup and intercall gaps separately; no outlier deletion or optional repeats based on the observed gain.",
                "exclude": ["sampling", "HTTP/SSE/UI", "finite scanning in T1 loop", "logit copying", "hashing", "export", "observer callbacks"],
                "statistics": "Report F/P separately for each case using total calls/total time and ms/token, paired repetition/block results and per-token tails; only2processes/mode. Do not treat768calls as768independent prompts. Do not present a general language effect from the translated pair.",
                "scope": "No C/S full timing grid, no bank reselection, no8layer candidate and no large-bank expansion in this stage.",
            },
            "free_answer_quality": {
                "modes": ["F", "P"], "responses_per_mode_per_case": 1, "total_responses": 6,
                "input": "Use execution_input.prefix_ids only. These saved IDs already include the frozen template, BOS and assistant boundary; do not template or tokenize them a second time. NEVER append the reference continuation when generating a free answer.",
                "sampler": {"policy": "greedy", "temperature": 0, "seed": 42,
                            "ignore_eos": False, "max_generated_tokens": 1024},
                "response_order": [{"case_id": SELECTED[0], "modes": ["F", "P"]},
                                   {"case_id": SELECTED[1], "modes": ["P", "F"]},
                                   {"case_id": SELECTED[2], "modes": ["F", "P"]}],
                "measurement_scope": "Task-answer check, not a balanced free-generation throughput benchmark; record native timing as descriptive only.",
                "capture": ["full output token IDs", "full decoded UTF8 output", "actual sampler/config", "stop reason", "generated count", "truncated flag", "elapsed and native timing with definitions"],
                "cap_policy": "If the1024-token cap is reached before a normal terminal condition, mark truncated=true and preserve the partial answer. Mark criteria requiring missing output incomplete, never imply a complete pass or silently rerun with a higher cap. Report truncation beside score.",
                "cap_counting": "The cap counts every generated token ID, including non-displayed reasoning/control IDs. Stop on the model's actual end-of-generation condition with ignore_eos=false and record the terminator. If a later higher-cap diagnostic is useful, give it a separate name, apply the same new cap to the F/P pair, and never replace the original six responses or imply that it was the initial test.",
                "grading": "Use the frozen rubrics below; retain each criterion pass/fail/incomplete and concrete evidence. Exact reference wording is not required. Do not tune35 from these responses; holdout remains unused.",
            },
        },
        "rubrics": rubrics,
        "integration_changes_required": [
            {"file": str(N / "integrated/driver.cpp"), "lines": [121, 123, 135],
             "change": "Select explicit case_id from this artifact and verify uniqueness. Replace tokens.size()==7, ci=2 and prefix==272 with the selected record's prefix_count and exact pinned IDs. The three valid prefixes are236,187,244; do not pad/truncate to272 or64."},
            {"file": str(N / "integrated/driver.cpp"), "lines": [123, 157, 158],
             "change": "The old cases[0] full_tokens identity guard and F_prefix_first4_equal_old byte gate are also old-wiring/old-JA-specific. Keep their provenance and apply them only to the matching old control; never compare new validation inputs/logits against those old cases/bytes. This is separation of existing guards, not a request for another old-JA test or removal of the new case's ID and numerical guards."},
            {"file": str(N / "integrated/driver.cpp"), "lines": [142, 143, 146],
             "change": "Store case_id/family/split/prefix hash/64inputs/65targets in every block. Include case_id,mode,quality-or-timing,repetition in artifact names; current quality-S.f32 etc would overwrite different cases."},
            {"file": str(N / "analyze_integrated.py"), "lines": [42, 48, 61, 83, 84],
             "change": "Join by explicit case_id and read the plan's dynamic prefix/IDs. Remove272/336/337 and cases[2] assumptions. Keep oldJA first5-row pin as a separately named old-wiring control; never compare new validation logits to oldJA bytes."},
            {"file": str(N / "integrated/run_integrated.py"), "lines": [63, 65],
             "change": "Freeze plan SHA, selected cases and runner/new-build hashes in launch receipt. ImplementF/P timing andS/P diagnosis explicitly; this JSON is not executable by the existing arm-only launcher as-is."},
            {"file": str(N / "integrated/driver.cpp"), "lines": [134, 139, 145],
             "change": "Use each case's true absolute positions. JA236..299 andPython244..307 cross256; EN187..250 does not in this64-window. Retain State/bank lifetime, completed mode-switch and graph rebuild checks; do not infer every256 position always rebuilds."},
        ],
        "decision_output": "Report all three cases, S/P equality, F/P latency, complete free answers and task rubrics; distinguish success, regression, truncation and inconclusive. This is a small validation screen, not automatic production adoption or the final holdout evaluation.",
    }
    plan_path = out / "FROZEN_VALIDATION3_PLAN.json"
    plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode()
    plan_path.write_bytes(plan_bytes)
    check = json.loads(plan_bytes)
    assert [c["case_id"] for c in check["cases"]] == SELECTED
    assert check["bank"]["plan_sha256"] == BANK_SHA
    # AST parsing validates this script and the supplied reference's syntax only; no exec/import.
    ast.parse(Path(__file__).read_text(), filename=str(Path(__file__)))
    receipt = {"kind": "offline_extraction_verification", "source_pin": PIN,
               "plan_file": plan_path.name, "plan_sha256": sha(plan_bytes), "plan_bytes": len(plan_bytes),
               "script_sha256": sha(Path(__file__).read_bytes()), "source_files": len(sources),
               "saved_RUN_json_reviewed": len(run_receipts), "cases": checks,
               "all_source_sha_and_i32_checks_passed": True,
               "json_roundtrip_passed": True, "script_ast_syntax_passed": True,
               "reference_python_ast_syntax_passed": True,
               "model_execution": False, "tokenizer_execution": False, "API_calls": False,
               "provided_scripts_executed": False, "provided_reference_code_executed": False,
               "unused_claim_scope": plan["unused_validation_evidence"]["claim"]}
    (out / "VALIDATION_PLAN_VERIFICATION.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
