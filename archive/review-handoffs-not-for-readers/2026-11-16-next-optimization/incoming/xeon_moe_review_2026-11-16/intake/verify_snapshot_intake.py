#!/usr/bin/env python3
"""Verify the pinned Git handoff, Release ZIP, and recorded runtime file hashes.

No archived code or ELF is executed. Optional ELF copies are written without
execute permission to a new directory, for static inspection only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

COMMIT = "7d8b077afcf1258602366245f1567e820255defa"
ZIP_SHA = "e4176a9dbe5dc6f252d23b2455edf8576c86d6e0376e830532c4988d5bd0253b"
RELATIVE = "archive/review-handoffs-not-for-readers/2026-11-16-current"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-elf-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.extract_elf_dir and args.extract_elf_dir.exists()):
        parser.error("Output file and optional extraction directory must be new")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if revision != COMMIT:
        parser.error("Unexpected Git commit")
    zip_hash = digest(args.zip.read_bytes())
    if zip_hash != ZIP_SHA:
        parser.error("Release ZIP SHA256 does not match the published value")
    shared = args.repo / RELATIVE
    manifest = json.loads((shared / "SHARED_TREE_MANIFEST.json").read_text())
    zip_manifest_bytes = (shared / "ZIP_CONTENTS_SHA256.json").read_bytes()
    zip_manifest = json.loads(zip_manifest_bytes)
    errors = []
    for entry in manifest["files"]:
        path = shared / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or digest(path.read_bytes()) != entry["sha256"]:
            errors.append({"shared_file_mismatch": entry["path"]})
    actual_shared = {str(path.relative_to(shared)) for path in shared.rglob("*") if path.is_file()}
    extra_shared = sorted(actual_shared - {entry["path"] for entry in manifest["files"]})
    if extra_shared != ["README.txt", "SHARED_TREE_MANIFEST.json"]:
        errors.append({"unexpected_extra_shared_files": extra_shared})
    binaries = json.loads((shared / "review/BINARY_SOURCE_MAP.json").read_text())["libraries"]
    elf_entries = []
    with zipfile.ZipFile(args.zip) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1 or len(names) != len(set(names)):
            parser.error("Unexpected ZIP root or duplicate member")
        prefix = next(iter(roots)) + "/"
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or not stat.S_ISREG(info.external_attr >> 16):
                parser.error("Unexpected ZIP member type or path")
        if archive.read(prefix + "CONTENTS_SHA256.json") != zip_manifest_bytes:
            errors.append({"zip_manifest_differs_from_git_copy": True})
        for entry in zip_manifest["files"]:
            content = archive.read(prefix + entry["path"])
            if len(content) != entry["size_bytes"] or digest(content) != entry["sha256"]:
                errors.append({"zip_file_mismatch": entry["path"]})
        expected = {prefix + entry["path"] for entry in zip_manifest["files"]} | {prefix + "CONTENTS_SHA256.json"}
        if set(names) != expected:
            errors.append({"zip_extra": sorted(set(names) - expected), "zip_missing": sorted(expected - set(names))})
        for row in binaries:
            packaged = row.get("packaged_runtime_path", "snapshot/llm-lab/" + row["runtime_path"].split("/llm-lab/", 1)[1])
            if digest(archive.read(prefix + packaged)) != row["runtime_sha256"]:
                errors.append({"runtime_map_mismatch": packaged})
        for row in manifest["ELF_omitted_from_git_available_in_release_zip"]:
            content = archive.read(prefix + row["path_in_zip"])
            if not content.startswith(b"\x7fELF") or digest(content) != row["sha256"]:
                errors.append({"elf_mismatch": row["path_in_zip"]})
            elf_entries.append({"path": row["path_in_zip"], "size_bytes": len(content), "sha256": digest(content)})
        if args.extract_elf_dir and not errors:
            args.extract_elf_dir.mkdir(parents=True, exist_ok=False)
            for entry in elf_entries:
                target = args.extract_elf_dir / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.read(prefix + entry["path"]))
                target.chmod(0o600)
    receipt = {
        "repo_commit": revision, "zip_sha256": zip_hash, "zip_bytes": args.zip.stat().st_size,
        "zip_root": prefix, "shared_files_verified": len(manifest["files"]),
        "extra_shared_publication_metadata": extra_shared,
        "zip_manifest_files_verified": len(zip_manifest["files"]), "zip_members": len(names),
        "runtime_file_hashes_verified": len(binaries), "elf_entries": elf_entries,
        "errors": errors, "all_matched": not errors,
        "scope": "File bytes and inventory only; no build, inference, library load or live-machine observation.",
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "elf_entries"}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
