#!/usr/bin/env python3
"""Record an existing Linux process's llama/ggml backing files without changing it.

No API calls, signals, model reads, builds, or subprocess commands are issued.
Hashes identify backing files, not relocated in-memory code or source revisions.
"""
import argparse
import hashlib
import json
import os
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DSO = re.compile(r"^(?:libggml|libllama|libmtmd)[-_A-Za-z0-9]*\.so(?:\.[A-Za-z0-9._-]+)?$")
SAFE_ENV = (
    "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS", "OMP_PROC_BIND", "OMP_PLACES",
    "OPENBLAS_NUM_THREADS", "GGML_CUDA_REGISTER_HOST", "GGML_OP_OFFLOAD_MIN_BATCH",
    "GGML_SCHED_MXFP4_OFFLOAD_MIN_BATCH", "GGML_CPU_IQ3_COMPACT_MIB",
)


def start_ticks(pid):
    value = Path(f"/proc/{pid}/stat").read_text()
    # comm may contain spaces or closing parentheses; its closing delimiter is last.
    fields = value[value.rfind(")") + 2:].split()
    return int(fields[19])  # stat field 22, counting from field 3 here.


def project_mappings(pid):
    groups = {}
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6:
            continue
        address, perms, offset, dev, inode, pathname = parts
        clean = pathname.removesuffix(" (deleted)")
        if not PROJECT_DSO.fullmatch(Path(clean).name):
            continue
        major, minor = (int(x, 16) for x in dev.split(":"))
        identity = (major, minor, int(inode))
        item = groups.setdefault(identity, {
            "device_major": major, "device_minor": minor, "inode": int(inode),
            "paths": [], "mappings": [],
        })
        if pathname not in item["paths"]:
            item["paths"].append(pathname)
        item["mappings"].append({"range": address, "permissions": perms, "offset_hex": offset})
    return sorted(groups.values(), key=lambda x: (x["paths"], x["inode"]))


def file_id(info):
    return (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)


def stat_signature(info):
    return (file_id(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def hash_open_file(path, expected, max_bytes):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not a regular backing file")
        if expected is not None and file_id(before) != expected:
            raise ValueError("path device/inode differs from the mapped file")
        if before.st_size > max_bytes:
            raise ValueError("file exceeds the configured hashing size limit")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ValueError("file grew beyond the hashing size limit")
            digest.update(block)
        after = os.fstat(fd)
        if stat_signature(before) != stat_signature(after) or total != before.st_size:
            raise ValueError("backing file changed while hashing")
        return {
            "status": "hashed", "sha256": digest.hexdigest(), "size_bytes": total,
            "device_major": os.major(before.st_dev), "device_minor": os.minor(before.st_dev),
            "inode": before.st_ino, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns, "path_used": str(path),
        }
    finally:
        os.close(fd)


def recheck_backing_file(record):
    """Recheck metadata after all hashes; do not accept replaced path identities."""
    if record["status"] != "hashed":
        return {"status": "unavailable", "error": "initial hash unavailable"}
    expected = ((record["device_major"], record["device_minor"], record["inode"]),
                record["size_bytes"], record["mtime_ns"], record["ctime_ns"])
    try:
        observed = os.stat(record["path_used"])
        if not stat.S_ISREG(observed.st_mode) or stat_signature(observed) != expected:
            return {"status": "changed", "observed_signature": stat_signature(observed)}
        return {"status": "unchanged"}
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc)}


def hash_mapped_group(pid, item, max_bytes):
    expected = (item["device_major"], item["device_minor"], item["inode"])
    attempts = []
    candidates = []
    for pathname in item["paths"]:
        clean = pathname.removesuffix(" (deleted)")
        if clean.startswith("/") and "\\012" not in clean and not pathname.endswith(" (deleted)"):
            # Resolve inside the target's mount namespace, not ours.
            candidates.append((f"/proc/{pid}/root{clean}", "target_mount_namespace_path"))
    for mapping in item["mappings"]:
        candidates.append((f"/proc/{pid}/map_files/{mapping['range']}", "mapped_file_handle"))
    for path, method in candidates:
        try:
            result = hash_open_file(path, expected, max_bytes)
            result.update({"method": method, "prior_failed_attempts": attempts})
            return result
        except (OSError, ValueError) as exc:
            attempts.append({"method": method, "path": path, "error": str(exc)})
    return {"status": "unavailable", "attempts": attempts}


def selected_environment(pid):
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        result = {}
        for part in raw.split(b"\0"):
            key, separator, value = part.partition(b"=")
            name = key.decode("ascii", "replace")
            if separator and name in SAFE_ENV:
                result[name] = value.decode("utf-8", "replace")
        return {"status": "read", "values": result}
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc)}


def collect(pid, max_bytes):
    before_start = start_ticks(pid)
    mappings = project_mappings(pid)
    try:
        exe_name = os.readlink(f"/proc/{pid}/exe")
        exe_hash = hash_open_file(f"/proc/{pid}/exe", None, max_bytes)
    except (OSError, ValueError) as exc:
        exe_name = None
        exe_hash = {"status": "unavailable", "error": str(exc)}
    files = [{**item, "backing_file": hash_mapped_group(pid, item, max_bytes)} for item in mappings]
    environment = selected_environment(pid)
    exe_hash["final_metadata_check"] = recheck_backing_file(exe_hash)
    for item in files:
        record = item["backing_file"]
        record["final_metadata_check"] = recheck_backing_file(record)
    metadata_stable = exe_hash["final_metadata_check"]["status"] == "unchanged" and all(
        item["backing_file"]["final_metadata_check"]["status"] == "unchanged" for item in files)
    try:
        after_start = start_ticks(pid)
        after_maps = project_mappings(pid)
        identity_stable = before_start == after_start
        mapping_stable = mappings == after_maps
        after_exe = os.stat(f"/proc/{pid}/exe")
        executable_stable = exe_hash["status"] == "hashed" and file_id(after_exe) == (
            exe_hash["device_major"], exe_hash["device_minor"], exe_hash["inode"])
        end_error = None
    except (OSError, ValueError, IndexError) as exc:
        after_start = None
        identity_stable = mapping_stable = executable_stable = False
        end_error = str(exc)
    complete = bool(files) and all(x["backing_file"]["status"] == "hashed" for x in files)
    verified = identity_stable and mapping_stable and executable_stable and metadata_stable and complete
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Existing process; libggml*, libllama*, libmtmd* shared-object backing files and executable only; no inference or configuration changes.",
        "host": {"kernel": os.uname().release, "machine": os.uname().machine},
        "process": {"pid": pid, "start_ticks_before": before_start, "start_ticks_after": after_start,
                    "executable_path": exe_name, "executable_backing_file": exe_hash},
        "selected_environment": environment,
        "project_libraries": files,
        "verification": {"process_identity_stable": identity_stable,
                         "project_mapping_snapshot_stable": mapping_stable,
                         "executable_identity_stable": executable_stable,
                         "backing_file_metadata_rechecked_stable": metadata_stable,
                         "project_library_hashes_complete": complete,
                         "verified_for_recorded_scope": verified, "end_error": end_error},
        "limits": [
            "A file hash does not identify its source commit or build flags; record those separately.",
            "Backing files are hashed, not relocated process memory; in-memory patches are not detected.",
            "Before/after snapshots can miss a transient unload/reload between observations.",
            "This is not an atomic snapshot: file metadata, mappings and process identity are observed sequentially. Changes after an individual final check may escape detection.",
            "CUDA/system DSOs, model weights, full command line and non-allowlisted environment are omitted.",
            "Environment records startup data and does not prove every setting's effective runtime value.",
            "A denied mapped-file handle is reported as unavailable; a replaced pathname is never accepted as the mapped file.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int, help="PID of an already running llama-server process")
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are not overwritten; default stdout")
    parser.add_argument("--max-file-mib", type=int, default=512)
    args = parser.parse_args()
    if args.pid <= 0 or args.max_file_mib <= 0:
        parser.error("pid and max-file-mib must be positive")
    try:
        data = collect(args.pid, args.max_file_mib * 1024 * 1024)
        rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as out:
                out.write(rendered)
        else:
            sys.stdout.write(rendered)
        return 0 if data["verification"]["verified_for_recorded_scope"] else 2
    except (OSError, ValueError, IndexError) as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
