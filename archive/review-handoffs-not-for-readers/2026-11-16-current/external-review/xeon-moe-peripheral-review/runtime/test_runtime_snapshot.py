#!/usr/bin/env python3
"""Exercise the read-only collector with local synthetic mapped files, not a model."""
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import tempfile
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("collector", HERE / "collect_runtime_snapshot.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
checks = []


def checked(name, condition, details=None):
    checks.append({"case": name, "passed": bool(condition), "details": details})
    assert condition, name


checked("project_library_names_cover_server_and_mtmd", all(collector.PROJECT_DSO.fullmatch(name) for name in (
    "libggml-base.so.0", "libllama.so.0", "libllama-common.so.0", "libllama-server-impl.so", "libmtmd.so.0"))
    and not collector.PROJECT_DSO.fullmatch("libcuda.so.1"))


with tempfile.TemporaryDirectory(prefix="runtime-review-") as temporary:
    folder = Path(temporary) / "path with spaces"
    folder.mkdir()
    path = folder / "libggml-review.so"
    old_bytes = b"original synthetic backing file\n" * 512
    new_bytes = b"replacement must not be misidentified\n" * 512
    path.write_bytes(old_bytes)
    with path.open("rb") as original:
        mapping = mmap.mmap(original.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            manifest = collector.collect(os.getpid(), 512 * 1024 * 1024)
            ours = [x for x in manifest["project_libraries"] if str(path) in x["paths"]]
            checked("mapped_file_with_spaces", len(ours) == 1 and ours[0]["backing_file"]["sha256"] == hashlib.sha256(old_bytes).hexdigest())
            checked("same_process_snapshot", manifest["verification"]["verified_for_recorded_scope"])

            select_environment = collector.selected_environment
            def mutate_after_hash(pid):
                path.write_bytes(b"M" * len(old_bytes))
                return select_environment(pid)
            with mock.patch.object(collector, "selected_environment", side_effect=mutate_after_hash):
                later_mutation = collector.collect(os.getpid(), 512 * 1024 * 1024)
            later_item = [x for x in later_mutation["project_libraries"] if str(path) in x["paths"]][0]
            checked("same_inode_change_after_hash_detected", later_item["backing_file"]["final_metadata_check"]["status"] == "changed"
                    and not later_mutation["verification"]["verified_for_recorded_scope"])
            path.write_bytes(old_bytes)

            replacement = folder / "replacement.tmp"
            replacement.write_bytes(new_bytes)
            os.replace(replacement, path)
            changed = collector.collect(os.getpid(), 512 * 1024 * 1024)
            deleted = [x for x in changed["project_libraries"] if str(path) + " (deleted)" in x["paths"]]
            result = deleted[0]["backing_file"] if deleted else {}
            checked("replaced_path_is_not_hashed_as_old_mapping", len(deleted) == 1 and result.get("sha256") != hashlib.sha256(new_bytes).hexdigest(), {"backing_file_status": result.get("status")})
            checked("deleted_mapping_is_original_or_unavailable", result.get("status") == "unavailable" or result.get("sha256") == hashlib.sha256(old_bytes).hexdigest())
            if result.get("status") == "unavailable":
                checked("unavailable_hash_makes_verification_incomplete", not changed["verification"]["verified_for_recorded_scope"])
        finally:
            mapping.close()

    info = path.stat()
    try:
        collector.hash_open_file(path, (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino + 1), 2**20)
    except ValueError:
        checked("wrong_inode_rejected", True)
    else:
        checked("wrong_inode_rejected", False)

    original_read = os.read
    mutated = [False]
    def mutating_read(fd, count):
        block = original_read(fd, count)
        if block and not mutated[0]:
            mutated[0] = True
            with path.open("ab") as target:
                target.write(b"changed during hashing")
        return block
    with mock.patch.object(collector.os, "read", side_effect=mutating_read):
        try:
            collector.hash_open_file(path, None, 2**20)
        except ValueError:
            checked("file_changed_during_hash_rejected", True)
        else:
            checked("file_changed_during_hash_rejected", False)

    with mock.patch.object(collector.Path, "read_bytes", return_value=b"LD_LIBRARY_PATH=/test\0API_KEY=do-not-export-this\0OMP_THREAD_LIMIT=12\0"):
        environment = collector.selected_environment(os.getpid())
    checked("environment_allowlist", environment["values"] == {"LD_LIBRARY_PATH": "/test", "OMP_THREAD_LIMIT": "12"} and "do-not-export-this" not in json.dumps(environment))

    with mock.patch.object(collector, "project_mappings", return_value=[]):
        empty = collector.collect(os.getpid(), 512 * 1024 * 1024)
    checked("empty_project_mapping_is_not_success", not empty["verification"]["verified_for_recorded_scope"])

    with mock.patch.object(collector, "start_ticks", side_effect=[100, 101]):
        reused = collector.collect(os.getpid(), 512 * 1024 * 1024)
    checked("pid_start_identity_change_detected", not reused["verification"]["process_identity_stable"])

    with mock.patch.object(collector, "project_mappings", side_effect=[[], [{"changed": True}]]):
        remapped = collector.collect(os.getpid(), 512 * 1024 * 1024)
    checked("mapping_snapshot_change_detected", not remapped["verification"]["project_mapping_snapshot_stable"])

result = {"scope": "Synthetic files mapped in this Python process; no shared library loaded as code, no engine or GPU execution.", "checks": checks, "all_passed": all(x["passed"] for x in checks)}
print(json.dumps(result, ensure_ascii=False, indent=2))
