import copy
import unittest

from analyze_moe_trace import analyze


def fixture():
    expected = [{"request": "synthetic", "pass": p, "phase": "decode"} for p in (10, 17)]
    records = []
    for p in expected:
        for layer in range(40):
            row = {**p, "layer": layer, "n_tokens": 1, "clock_domain": "synthetic-process",
                   "route": [(layer * 7 + slot) % 384 for slot in range(6)],
                   "backend": "cpu", "gpu_hit_mask": 0, "host_ns": None,
                   "gpu_stream_ms": None}
            if layer in (0, 1, 2):
                h = (0, 2, 6)[layer]
                row.update(backend="hot", gpu_hit_mask=(1 << h) - 1)
                if h == 0:
                    times = {"wall": [0, 15000000], "cpu_setup": [1000000, 2000000],
                             "cpu_compute": [2000000, 12000000], "cpu_output_copy": [12000000, 13000000]}
                elif h == 6:
                    times = {"wall": [0, 10000000], "gpu_submit_host": [1000000, 2000000],
                             "gpu_join": [2000000, 8000000], "scatter": [8000000, 9000000]}
                else:
                    times = {"wall": [0, 16000000], "cpu_setup": [1000000, 2000000],
                             "gpu_submit_host": [2000000, 3000000], "cpu_compute": [3000000, 9000000],
                             "cpu_output_copy": [9000000, 10000000], "gpu_join": [10000000, 14000000],
                             "scatter": [14000000, 15000000]}
                    row.update(gpu_done_query="not_ready", gpu_done_query_host_ns=10000000)
                row["host_ns"] = times
                if h:
                    row.update(gpu_stream_ms={"h2d": .2, "graph": 5, "d2h": .3}, gpu_events_completed=True)
            elif layer >= 37:
                row.update(backend="gpu", gpu_hit_mask=63)
            records.append(row)
    return {"schema_version": 1, "synthetic_only": True,
            "manifest": {"complete": True, "dropped_records": 0, "layers": 40, "top_k": 6, "experts": 384,
                         "host_clock": "CLOCK_MONOTONIC_RAW", "clock_domain": "synthetic-process",
                         "config_sha256": "0" * 64, "expected_passes": expected}, "records": records}


class TraceTests(unittest.TestCase):
    def test_hot_only_scope_is_not_a_40_layer_census(self):
        data = fixture()
        data["manifest"].update(trace_kind="hot_latency", expected_layers=[0, 1, 2])
        data["records"] = [r for r in data["records"] if r["backend"] == "hot"]
        result = analyze(data)
        self.assertEqual(result["records"], 6)
        self.assertFalse(result["is_complete_40_layer_census"])
        data["records"] = [r for r in data["records"] if r["layer"] != 2]
        with self.assertRaises(ValueError): analyze(data)

    def test_hot_latency_cannot_omit_all_host_times(self):
        data = fixture()
        data["manifest"].update(trace_kind="hot_latency", expected_layers=[0, 1, 2])
        data["records"] = [r for r in data["records"] if r["backend"] == "hot"]
        for row in data["records"]:
            row["host_ns"] = row["gpu_stream_ms"] = None
        with self.assertRaises(ValueError): analyze(data)

    def test_complete_and_separate_time_domains(self):
        result = analyze(fixture())
        self.assertEqual(result["records"], 80)
        self.assertEqual(result["hot_host_interval_ms"]["wall"]["n"], 6)
        self.assertEqual(result["gpu_stream_elapsed_ms"]["graph"]["n"], 4)
        self.assertAlmostEqual(result["exposed_join_fraction_of_observed_hot_wall"], 20 / 82)
        self.assertEqual(result["layers"][37]["hot_dispatcher_H_histogram"], [0] * 7)
        self.assertEqual(result["layers"][37]["current_gpu_slot_H_histogram"][6], 2)

    def test_route_only_does_not_manufacture_zero_timings(self):
        data = fixture()
        for row in data["records"]:
            row["host_ns"] = row["gpu_stream_ms"] = None
        result = analyze(data)
        self.assertEqual(result["hot_host_interval_ms"], {})
        self.assertIsNone(result["exposed_join_fraction_of_observed_hot_wall"])

    def test_missing_same_layer_every_pass_rejected(self):
        data = fixture(); data["records"] = [r for r in data["records"] if r["layer"] != 39]
        with self.assertRaises(ValueError): analyze(data)

    def test_duplicate_replacing_missing_rejected(self):
        data = fixture(); data["records"][-1] = copy.deepcopy(data["records"][0])
        with self.assertRaises(ValueError): analyze(data)

    def test_empty_expected_or_dropped_rejected(self):
        for mode in ("empty", "drop", "incomplete"):
            with self.subTest(mode=mode):
                data = fixture()
                if mode == "empty": data["manifest"]["expected_passes"] = []; data["records"] = []
                if mode == "drop": data["manifest"]["dropped_records"] = 1
                if mode == "incomplete": data["manifest"]["complete"] = False
                with self.assertRaises(ValueError): analyze(data)

    def test_bad_route_clock_and_shape_rejected(self):
        for field, value in (("route", [0, 0, 2, 3, 4, 5]), ("route", [0, 1, 2, 3, 4, 384]),
                             ("n_tokens", 2), ("clock_domain", "another-process")):
            with self.subTest(field=field, value=value):
                data = fixture(); data["records"][0][field] = value
                with self.assertRaises(ValueError): analyze(data)

    def test_overlap_missing_branch_and_reversal_rejected(self):
        for name, value in (("gpu_join", [9000000, 14000000]), ("cpu_compute", None), ("scatter", [16000000, 15000000])):
            with self.subTest(name=name):
                data = fixture(); data["records"][1]["host_ns"][name] = value
                with self.assertRaises(ValueError): analyze(data)

    def test_gpu_nonfinite_or_uncompleted_rejected(self):
        for mode in ("nan", "uncompleted", "query"):
            with self.subTest(mode=mode):
                data = fixture(); row = data["records"][1]
                if mode == "nan": row["gpu_stream_ms"]["graph"] = float("nan")
                if mode == "uncompleted": row["gpu_events_completed"] = False
                if mode == "query": row["gpu_done_query_host_ns"] = 1
                with self.assertRaises(ValueError): analyze(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
