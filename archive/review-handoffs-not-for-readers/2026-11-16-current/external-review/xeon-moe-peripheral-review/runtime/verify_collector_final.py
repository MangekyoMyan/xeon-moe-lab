#!/usr/bin/env python3
"""Independently verify collector fixes using data-only synthetic mmap fixtures."""
import argparse
import hashlib
import importlib.util
import json
import mmap
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--previous-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    collector_sha = digest(args.collector)
    previous_sha = digest(args.previous_review)
    spec = importlib.util.spec_from_file_location("independent_collector_final", args.collector)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    checks = {}
    expected_names = [
        "libggml-base.so.0.28.0", "libggml-cuda.so", "libllama.so.0.8.0",
        "libllama-common.so.0.8.0", "libllama-server-impl.so", "libmtmd.so.0.8.0",
    ]
    unrelated_names = ["libc.so.6", "libcuda.so.1", "notlibllama.so", "libllama.so.tmp/other"]
    matches = {name: bool(collector.PROJECT_DSO.fullmatch(name))
               for name in expected_names + unrelated_names}
    checks["project_dso_names_included"] = all(matches[name] for name in expected_names)
    checks["unrelated_names_excluded"] = not any(matches[name] for name in unrelated_names)
    cases = {}
    original_env = collector.selected_environment
    old = b"a" * 16384
    new = b"b" * 16384

    with tempfile.TemporaryDirectory(prefix="collector-final-independent-") as directory:
        path = Path(directory) / "libggml-independent.so"
        for case in ("unchanged", "post_hash_same_inode_overwrite", "post_hash_overwrite_mtime_restored"):
            path.write_bytes(old)
            initial_stat = path.stat()
            with path.open("rb") as source:
                mapping = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
                try:
                    def after_hash(pid):
                        if case != "unchanged":
                            path.write_bytes(new)
                            if case == "post_hash_overwrite_mtime_restored":
                                os.utime(path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
                        return original_env(pid)

                    with mock.patch.object(collector, "selected_environment", side_effect=after_hash):
                        snapshot = collector.collect(os.getpid(), 512 * 1024 * 1024)
                    item = next(item for item in snapshot["project_libraries"] if str(path) in item["paths"])
                    backing = item["backing_file"]
                    final_stat = path.stat()
                    final_sha = digest(path)
                    stable_identity = (initial_stat.st_dev, initial_stat.st_ino) == (final_stat.st_dev, final_stat.st_ino)
                    changed = case != "unchanged"
                    cases[case] = {
                        "verification": snapshot["verification"],
                        "backing_file": backing,
                        "final_sha256": final_sha,
                        "same_device_and_inode": stable_identity,
                        "mtime_unchanged": initial_stat.st_mtime_ns == final_stat.st_mtime_ns,
                        "ctime_unchanged": initial_stat.st_ctime_ns == final_stat.st_ctime_ns,
                    }
                    checks[case + ":recorded_original_hash"] = backing.get("sha256") == hashlib.sha256(old).hexdigest()
                    checks[case + ":same_inode"] = stable_identity
                    checks[case + ":expected_final_hash"] = final_sha == hashlib.sha256(new if changed else old).hexdigest()
                    checks[case + ":expected_verification"] = snapshot["verification"]["verified_for_recorded_scope"] is (not changed)
                    checks[case + ":expected_metadata_result"] = backing["final_metadata_check"]["status"] == ("changed" if changed else "unchanged")
                    checks[case + ":pid_maps_exe_still_stable"] = all(
                        snapshot["verification"][key] for key in (
                            "process_identity_stable", "project_mapping_snapshot_stable", "executable_identity_stable"))
                    if case == "post_hash_overwrite_mtime_restored":
                        checks[case + ":mtime_restored"] = initial_stat.st_mtime_ns == final_stat.st_mtime_ns
                        checks[case + ":ctime_changed"] = initial_stat.st_ctime_ns != final_stat.st_ctime_ns
                finally:
                    mapping.close()

    checks["collector_unchanged_during_verification"] = digest(args.collector) == collector_sha
    checks["earlier_review_preserved"] = digest(args.previous_review) == previous_sha
    result = {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent synthetic data mmap only; no ELF loaded as executable code, no service/model/GPU operation, no repository edits.",
        "collector_path": str(args.collector),
        "collector_sha256": collector_sha,
        "previous_review_path": str(args.previous_review),
        "previous_review_sha256": previous_sha,
        "project_name_matches": matches,
        "cases": cases,
        "checks": checks,
        "all_passed": all(checks.values()),
        "limits": [
            "Verifies the reported post-hash same-inode overwrite and DSO-name omissions only, with one unchanged control.",
            "The collector remains a sequential observation, not an atomic snapshot; a mutation after an individual final metadata check can escape detection.",
            "Does not establish source revision, build flags, relocated/in-memory bytes, or the user's live runtime configuration.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "collector_sha256": collector_sha,
                      "all_passed": result["all_passed"], "checks": len(checks),
                      "failed_checks": [name for name, passed in checks.items() if not passed]}, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
