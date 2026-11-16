#!/usr/bin/env python3
"""Offline failure/identity/boundary checks. nvidia-smi is mocked in every CLI check."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("snapshot_capacity.py")
spec = importlib.util.spec_from_file_location("capacity", SOURCE)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)
UUID = "GPU-11111111-2222-3333-4444-555555555555"


def xml(free="2848 MiB", used="29920 MiB"):
    return f'''<nvidia_smi_log><driver_version>fixture</driver_version><gpu id="fixture">
      <uuid>{UUID}</uuid><product_name>offline fixture</product_name>
      <fb_memory_usage><total>32768 MiB</total><used>{used}</used><free>{free}</free></fb_memory_usage>
      <gpu_power_readings><current_power_limit>150.00 W</current_power_limit></gpu_power_readings>
      <ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode>
    </gpu></nvidia_smi_log>'''.encode()


class OfflineChecks(unittest.TestCase):
    def test_threshold_strictness_and_no_authorization(self):
        for extra, expected in [(-1, False), (0, False), (1, True)]:
            d = capacity.screen(3072 * capacity.MIB + 19 + extra, 19)
            self.assertIs(d["reported_free_passes_candidate_memory_inequality"], expected)
            self.assertIs(d["runtime_admission_or_launch_authorized"], False)
            self.assertIs(d["postallocation_guard_evaluated"], False)

    def test_unknown_and_zero_payload_never_become_admission(self):
        self.assertIsNone(capacity.screen(None, 19)["reported_free_passes_candidate_memory_inequality"])
        self.assertIsNone(capacity.screen(4000 * capacity.MIB, None)["reported_free_passes_candidate_memory_inequality"])
        zero = capacity.screen(0, 0)
        self.assertTrue(zero["zero_payload_skips_this_positive_payload_preallocation_guard"])
        self.assertIsNone(zero["reported_free_passes_candidate_memory_inequality"])
        self.assertFalse(zero["runtime_admission_or_launch_authorized"])

    def test_identity_and_single_gpu_are_required(self):
        with self.assertRaises(ValueError):
            capacity.parse_gpu_xml(xml(), "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        twice = b"<nvidia_smi_log>" + b"<gpu><uuid>x</uuid></gpu>" * 2 + b"</nvidia_smi_log>"
        with self.assertRaises(ValueError):
            capacity.parse_gpu_xml(twice, "0")

    def test_unknown_units_and_impossible_values(self):
        self.assertIsNone(capacity.parse_gpu_xml(xml("N/A"), UUID)["memory"]["free_bytes"])
        for bad in ["4096 MB", "NaN MiB", "Infinity MiB", "-1 MiB", "40000 MiB"]:
            with self.assertRaises(ValueError):
                capacity.parse_gpu_xml(xml(bad), UUID)

    def test_read_only_cli_and_unknown_exit(self):
        for data, returncode, expected_exit in [(xml(), 0, 0), (xml("N/A"), 0, 2),
                                               (b"malformed", 0, 2), (b"", 1, 2)]:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "receipt.json"
                fake = subprocess.CompletedProcess([], returncode, data, b"fixture")
                args = [str(SOURCE), "--gpu", UUID, "--out", str(out)]
                with patch.object(sys, "argv", args), patch.object(capacity.subprocess, "run", return_value=fake) as run, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(capacity.main(), expected_exit)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0], ["nvidia-smi", "--id=" + UUID, "-q", "-x"])
                self.assertEqual(run.call_args.kwargs["timeout"], 15)
                receipt = json.loads(out.read_text())
                self.assertFalse(receipt["capacity_screen"]["runtime_admission_or_launch_authorized"])
                self.assertEqual(receipt["engine_or_API_actions"], [])

    def test_existing_receipt_is_preserved_without_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            out.write_text("original")
            with patch.object(sys, "argv", [str(SOURCE), "--gpu", UUID, "--out", str(out)]), patch.object(capacity.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    capacity.main()
                run.assert_not_called()
            self.assertEqual(out.read_text(), "original")

    def test_missing_host_available_is_explicitly_unknown(self):
        with patch.object(capacity.Path, "read_text", return_value="MemTotal: 123 kB\n"):
            result = capacity.host_memory()
        self.assertIsNone(result["values"]["MemAvailable_bytes"])
        self.assertEqual(result["missing_fields"], ["MemAvailable"])
        self.assertFalse(result["complete"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OfflineChecks)
    test_names = [test.id().split(".")[-1] for test in suite]
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    receipt = {"scope": "Offline fixtures; all nvidia-smi calls mocked; no real GPU query or model run.",
               "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "tests": test_names, "tests_run": result.testsRun, "passed": result.wasSuccessful()}
    Path(__file__).with_name("CAPACITY_OFFLINE_CHECKS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
