#!/usr/bin/env python3
"""SYNTHETIC analyzer tests. No model, engine, ELF, or performance benchmark runs."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import analyze_mixed_run as audit


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_fixture(root, outlier=False):
    root.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(audit.SCHEDULE):
        folder = root / f"process-{index}"; folder.mkdir()
        cpu = "/SYNTHETIC/" + ("frozen" if label == "F" else "overlay") + "/libggml-cpu.so"
        meta = {"kind": "mixed_meta", "schema": 1, "synthetic": True,
                "process_index": index, "library_label": label, "commit": audit.COMMIT,
                "tensor": "blk.0.ffn_up_exps.weight", "threads": 12, "blocks": 16, "repeats": 32,
                "routes": list(audit.ROUTES), "case_order": audit.case_order_for(index),
                "inputs_sha256": audit.INPUT_SHA, "plan_sha256": audit.PLAN_SHA,
                "cpu_provider": cpu, "base_provider": "/SYNTHETIC/common/libggml-base.so",
                "cache_provider": cpu, "mode_provider": cpu if label == "E" else "", "ggml_numa": False,
                "clock": "CLOCK_MONOTONIC", "hashes_verified_by_harness": False}
        records = [meta]; start = 1_000_000_000 + index * 1_000_000_000_000; clock = start + 10000
        for case in audit.case_order_for(index):
            selected = audit.selected_for(case); masks = audit.masks_for(selected)
            records.append({"kind": "mixed_case", "case": case, "cache_bytes": len(selected) * audit.EXPERT_BYTES,
                            "selected_ids": selected, "masks": masks, "pins": list(range(12)),
                            "expected_file": f"expected-{case}.bin", "expected_groups": 8,
                            "expected_values_per_group": audit.OUTPUT_VALUES_PER_GROUP,
                            "warmup_arm_calls_per_group": 4, "bit_mismatches": 0, "dispatch_checked": label == "E"})
            # Cases deliberately have different expected bytes: the contract
            # checks equality across processes within a case, not across cases.
            blob = struct.pack("<f", 1.0 + audit.CASES.index(case)) * (audit.OUTPUT_BYTES_PER_CASE // 4)
            (folder / f"expected-{case}.bin").write_bytes(blob)
            for block in range(16):
                group = block % 8
                for arm, logical in enumerate(audit.logical_order(block)):
                    actual = int(label == "E" and logical == "S")
                    elapsed = 2_880_000 if actual else 3_200_000  # Artificial clocks, not measurements.
                    if outlier and index == 1 and case == "C0" and block == 0 and arm == 1:
                        elapsed *= 100
                    begin = clock; end = begin + elapsed; clock = end + 1000
                    records.append({"kind": "mixed_chunk", "case": case, "block": block, "route_group": group,
                                    "arm": arm, "logical_mode": logical, "actual_mode": actual,
                                    "cached_of_six": masks[group].bit_count(), "hit_mask": masks[group],
                                    "ids": list(audit.ROUTES[6 * group:6 * group + 6]),
                                    "begin_ns": begin, "end_ns": end, "elapsed_ns": elapsed, "repeats": 32,
                                    "ms_per_graph": elapsed / (32 * 1e6), "bit_mismatches": 0})
        records.append({"kind": "mixed_complete", "passed": True, "chunks": 192, "timed_graph_calls": 6144,
                        "expected_output_scope": "mode0 once per case/group; final output of each timed chunk compared in memory",
                        "outputs_bytes": 3 * audit.OUTPUT_BYTES_PER_CASE})
        (folder / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        status = {"schema": 1, "synthetic": True, "exit_code": 0, "error": None, "passed": True, "state": "completed",
                  "process_index": index, "library_label": label, "files": {},
                  "identity": {"cpu_library_sha256": audit.CPU_SHA[label], "base_library_sha256": "a" * 64,
                               "harness_sha256": "b" * 64, "inputs_sha256": audit.INPUT_SHA, "plan_sha256": audit.PLAN_SHA,
                               "cpu_library_path": cpu, "base_library_path": "/SYNTHETIC/common/libggml-base.so",
                               "case_spec_sha256": "c" * 64, "case_header_sha256": "d" * 64, "launcher_sha256": "e" * 64,
                               "shard_stat": {"path": "/SYNTHETIC/model.gguf", "device": 1, "inode": 2, "bytes": 123456, "mtime_ns": 1000000000},
                               "extra_library_dirs": ["/SYNTHETIC/extra"]},
                  "started_monotonic_ns": start, "ended_monotonic_ns": clock + 10000}
        for name in ("raw.jsonl", "expected-C0.bin", "expected-C6.bin", "expected-Cplan.bin"):
            data = (folder / name).read_bytes(); status["files"][name] = {"bytes": len(data), "sha256": audit.digest(data)}
        save_json(folder / "STATUS.json", status)


def edit_raw(root, index, change):
    folder = root / f"process-{index}"; path = folder / "raw.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    refresh_file(root, index, "raw.jsonl")


def refresh_file(root, index, name):
    folder = root / f"process-{index}"; status = load_json(folder / "STATUS.json")
    data = (folder / name).read_bytes(); status["files"][name] = {"bytes": len(data), "sha256": audit.digest(data)}
    save_json(folder / "STATUS.json", status)


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC_mixed_analyzer_")
        self.root = Path(self.temporary.name) / "run"
        make_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_rejected(self, fragment):
        with self.assertRaisesRegex(audit.EvidenceError, fragment):
            audit.analyze_run(self.root)

    def test_valid_normalization_and_process_units(self):
        result = audit.analyze_run(self.root)
        self.assertTrue(result["passed"])
        self.assertIn("SYNTHETIC", result["input_scope"])
        self.assertEqual(result["checks"]["chunks"], 1536)
        self.assertEqual(result["checks"]["timed_graph_calls"], 49152)
        self.assertEqual(len(result["primary_overlay_within_process"]), 12)
        for row in result["primary_overlay_within_process"]:
            self.assertAlmostEqual(row["ratio_of_total_times"], 10 / 9)
            self.assertEqual(row["timed_graph_calls_each_logical_mode"], 1024)
        for row in result["frozen_logical_placebo"]:
            self.assertEqual(row["ratio_of_total_times"], 1)
        for row in result["cross_library_pairs"]:
            self.assertAlmostEqual(row["ratio_of_normalized_total_times"], 1 if row["overlay_logical_mode"] == "O" else 10 / 9)
            self.assertEqual(row["frozen_raw_graph_calls"], 2 * row["overlay_logical_graph_calls"])
        for row in result["descriptive_summaries"]["overlay_within_process"]:
            self.assertEqual(row["process_replicate_count"], 4)
            self.assertAlmostEqual(row["pooled_block_paired_median"], 10 / 9)
            self.assertEqual(row["pooled_block_ratio_count"], 64)
        for row in result["descriptive_summaries"]["frozen_placebo"]:
            self.assertEqual(row["pooled_block_paired_median"], 1)
            self.assertEqual(row["pooled_block_ratio_count"], 64)
        for row in result["descriptive_summaries"]["cross_library"]:
            self.assertAlmostEqual(row["pooled_block_paired_median"], 1 if row["overlay_logical_mode"] == "O" else 10 / 9)
            self.assertEqual(row["pooled_block_ratio_count"], 64)

    def test_outlier_retained_and_estimands_separated(self):
        other = Path(self.temporary.name) / "outlier"; make_fixture(other, outlier=True)
        result = audit.analyze_run(other)
        row = next(r for r in result["primary_overlay_within_process"] if r["process_index"] == 1 and r["case"] == "C0")
        self.assertAlmostEqual(row["block_paired_ratios"]["median"], 10 / 9)
        self.assertAlmostEqual(row["ratio_of_total_times"], (32 * 3_200_000) / (131 * 2_880_000))
        self.assertEqual(row["block_paired_ratios"]["n"], 16)
        self.assertAlmostEqual(row["logical_S_chunk_ms"]["max"], 9.0)

    def test_missing_process(self):
        shutil.rmtree(self.root / "process-7")
        self.assert_rejected("exactly process-0")

    def test_incomplete_raw(self):
        edit_raw(self.root, 1, lambda rows: rows.pop())
        self.assert_rejected("missing/failed mixed_complete")

    def test_failed_completion_and_cli_failed_report(self):
        edit_raw(self.root, 1, lambda rows: rows[-1].update(passed=False))
        before = (self.root / "process-1/raw.jsonl").read_bytes()
        report = Path(self.temporary.name) / "FAILED_REPORT.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["--run-root", str(self.root), "--output", str(report)])
        self.assertEqual(code, 1)
        self.assertFalse(load_json(report)["passed"])
        self.assertTrue(load_json(report)["evidence_retained"])
        self.assertEqual(before, (self.root / "process-1/raw.jsonl").read_bytes())

    def test_duplicate_chunk(self):
        edit_raw(self.root, 1, lambda rows: rows.__setitem__(3, dict(rows[2])))
        self.assert_rejected("arm")

    def test_bad_elapsed(self):
        edit_raw(self.root, 1, lambda rows: rows[2].update(elapsed_ns=rows[2]["elapsed_ns"] + 1))
        self.assert_rejected("elapsed mismatch")

    def test_bad_mean_time(self):
        edit_raw(self.root, 1, lambda rows: rows[2].update(ms_per_graph=999.0))
        self.assert_rejected("ms/graph mismatch")

    def test_nonfinite_number(self):
        edit_raw(self.root, 1, lambda rows: rows[2].update(ms_per_graph=float("nan")))
        self.assert_rejected("nonfinite")

    def test_actual_mode_wrong_on_frozen(self):
        edit_raw(self.root, 0, lambda rows: rows[3].update(actual_mode=1))
        self.assert_rejected("actual mode")

    def test_output_bytes_differ_despite_valid_file_receipt(self):
        path = self.root / "process-7/expected-Cplan.bin"
        data = bytearray(path.read_bytes()); data[:4] = struct.pack("<f", 7.0); path.write_bytes(data)
        refresh_file(self.root, 7, "expected-Cplan.bin")
        self.assert_rejected("Saved expected bytes differ")

    def test_cpu_library_identity(self):
        path = self.root / "process-1/STATUS.json"; status = load_json(path)
        status["identity"]["cpu_library_sha256"] = "c" * 64; save_json(path, status)
        self.assert_rejected("unexpected CPU library hash")

    def test_common_harness_identity(self):
        path = self.root / "process-3/STATUS.json"; status = load_json(path)
        status["identity"]["harness_sha256"] = "c" * 64; save_json(path, status)
        self.assert_rejected("Common identity differs")

    def test_outer_process_overlap(self):
        p0 = load_json(self.root / "process-0/STATUS.json")
        path = self.root / "process-1/STATUS.json"; status = load_json(path)
        status["started_monotonic_ns"] = p0["ended_monotonic_ns"] - 1; save_json(path, status)
        self.assert_rejected("Outer processes overlap")

    def test_wrong_cache_mask(self):
        edit_raw(self.root, 0, lambda rows: rows[1].update(masks=[1] * 8))
        self.assert_rejected("masks")

    def test_missing_required_status_error_field(self):
        path = self.root / "process-1/STATUS.json"; status = load_json(path); del status["error"]; save_json(path, status)
        self.assert_rejected("unsuccessful/partial STATUS")

    def test_false_status_passed_despite_exit_zero(self):
        path = self.root / "process-1/STATUS.json"; status = load_json(path)
        status["passed"] = False; save_json(path, status)
        self.assert_rejected("unsuccessful/partial STATUS")

    def test_provider_path_mismatch(self):
        path = self.root / "process-1/STATUS.json"; status = load_json(path)
        status["identity"]["cpu_library_path"] = "/SYNTHETIC/wrong/libggml-cpu.so"
        save_json(path, status)
        self.assert_rejected("provider path differs from launcher identity")

    def test_shard_stat_changes_across_processes(self):
        path = self.root / "process-3/STATUS.json"; status = load_json(path)
        status["identity"]["shard_stat"]["mtime_ns"] += 1; save_json(path, status)
        self.assert_rejected("Common identity differs: shard_stat")

    def test_extra_library_directories_change(self):
        path = self.root / "process-3/STATUS.json"; status = load_json(path)
        status["identity"]["extra_library_dirs"].append("/SYNTHETIC/different-extra"); save_json(path, status)
        self.assert_rejected("Common identity differs: extra_library_dirs")

    def test_file_hash_mismatch(self):
        path = self.root / "process-1/raw.jsonl"; path.write_bytes(path.read_bytes() + b" ")
        self.assert_rejected("file hash/length mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "MIXED_ANALYZER_TEST_RESULT.json")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AnalyzerTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"scope": "SYNTHETIC OFFLINE ANALYZER TESTS; NOT PERFORMANCE EVIDENCE",
              "passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "performance_benchmarks_run": 0, "model_or_elf_executions": 0,
              "analyzer_sha256": audit.digest(Path(audit.__file__).read_bytes()),
              "test_source_sha256": audit.digest(Path(__file__).read_bytes()), "log": stream.getvalue()}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2); output.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}))
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
