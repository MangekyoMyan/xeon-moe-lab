"""Freeze selected source/evidence and recorded runtime bytes for external review."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
PUBLIC = LAB.parent / "xeon-moe-lab"
STAGE = ROOT / "stage/xeon-moe-current-20261116"
TEXT = {".py", ".c", ".cpp", ".cc", ".h", ".hpp", ".cuh", ".cu", ".inc", ".patch", ".json", ".txt", ".csv", ".plan", ".log", ".tsv", ".md", ".cmake", ".sh", ".in", ".template", ".service"}
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "models", "audit-results"}
EXCLUDE_NAMES = {"AGENTS.md", "RESUME.txt", "STATE.json", "ADVISOR_NOTES.txt", "server.stderr.log", "server.stdout.log", "health.json", "server-config.json", "baseline-config.private.json", "api-key.txt"}
SECRET_KEYS = {"api_key", "api-key", "authorization", "access_token", "refresh_token", "password", "secret", "api_key_file"}
manifest = []
omitted = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {k: "<REDACTED>" if k.lower() in SECRET_KEYS and v is not None else clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(x) for x in value]
    return value


def add(source, target, expected_sha=None):
    source = Path(source)
    target = Path(target)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("Unsafe destination")
    dst = STAGE / target
    if dst.exists():
        return
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("Source changed during copy: " + str(source))
    original = sha(raw)
    if expected_sha and original != expected_sha:
        raise RuntimeError("Recorded binary hash mismatch: " + str(source))
    shared = raw
    if source.suffix == ".json":
        try:
            value = json.loads(raw)
            cleaned = clean_json(value)
            if cleaned != value:
                shared = (json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n").encode()
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(shared)
    dst.chmod(0o644)
    manifest.append({"file": str(target), "source": str(source), "source_sha256": original,
                     "sha256": sha(shared), "size_bytes": len(shared), "redacted": raw != shared,
                     "source_mtime_ns": before.st_mtime_ns})


def tree(source, target, source_tree=False):
    source = Path(source)
    for base, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".git"))
        for name in sorted(files):
            p = Path(base) / name
            rel = p.relative_to(source)
            if p.is_symlink() or name in EXCLUDE_NAMES or ".private." in name or "profiles" in rel.parts or name.startswith("RESUME"):
                continue
            if source_tree:
                keep = p.suffix not in {".gguf", ".bin", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".pdf", ".zip", ".gz", ".parquet", ".safetensors"}
            else:
                keep = p.suffix in TEXT and not name.endswith(".so")
            if not keep:
                continue
            if p.stat().st_size > 16 * 1024**2:
                omitted.append({"source": str(p), "reason": "large text artifact; not required for selected comparisons", "size_bytes": p.stat().st_size})
                continue
            add(p, Path(target) / rel)


def main():
    if STAGE.exists():
        raise RuntimeError("Choose a fresh staging directory; do not overwrite a frozen snapshot")
    STAGE.mkdir(parents=True)
    for name in ["CURRENT_NOTE_JA.txt", "ROOT_TRIAGE_JA.txt", "TRIAGE_TERRA.txt", "INCOMING_RECEIPT.json", "RUNTIME_SNAPSHOT.json", "BINARY_SOURCE_MAP.json", "capture_current.py", "package_snapshot.py"]:
        add(ROOT / name, Path("review") / name)
    tree(ROOT / "capture", "review/capture")
    tree(ROOT / "incoming", "external-review")
    for rel in ["deepseek41-bringup-20261101/source", "mxfp4-hybrid-20261102/hybrid/engine"]:
        tree(LAB / rel, Path("snapshot/llm-lab") / rel, source_tree=True)
    for rel in ["exploration-20261112", "optimization-20261112", "iq-cache-next-20261103", "iq3-cache-20261103", "gpu-stream-20261103"]:
        tree(LAB / rel, Path("snapshot/llm-lab") / rel)
    operational = {
        "deepseek41-bringup-20261101": ["start.py", "supervise.py", "gpu_guard.py", "status.py", "build.py", "build-commands.json", "BUILD_READY.json", "README_JA.txt", "llm-deepseek41-api.service"],
        "thermal-monitor-20261101": ["monitor.py"],
        "moe-engine-20261029": ["watch.py"],
        "board-study-20261024": ["hardware_run.py"],
        "review-verification-20261101": ["model-tensor-index.json"],
    }
    for directory, files in operational.items():
        for name in files:
            p = LAB / directory / name
            if p.exists():
                add(p, Path("snapshot/llm-lab") / directory / name)
    for name in ["bench_deepseek_light.py", "glm_q8_experiment.py"]:
        add(LAB / name, Path("snapshot/llm-lab") / name)
    for rel in ["deepseek41-bringup-20261101/build/compile_commands.json", "mxfp4-hybrid-20261102/hybrid/build-cpu-ivybridge/compile_commands.json"]:
        add(LAB / rel, Path("snapshot/llm-lab") / rel)
    # These exact files are the observed executable/libraries; never scan models or toolkit binaries.
    runtime = json.loads((ROOT / "RUNTIME_SNAPSHOT.json").read_text())
    if not runtime["verification"]["verified_for_recorded_scope"]:
        raise RuntimeError("Incomplete runtime observation")
    runtime_rows = [(runtime["process"]["executable_path"], runtime["process"]["executable_backing_file"])]
    runtime_rows += [(x["paths"][0], x["backing_file"]) for x in runtime["project_libraries"]]
    for original, record in runtime_rows:
        p = Path(original)
        add(p, Path("snapshot/llm-lab") / p.relative_to(LAB), record["sha256"])
    hot = LAB / "exploration-20261112/hot-ffn/release-a/libllama.so.0.8.0"
    add(hot, Path("snapshot/llm-lab") / hot.relative_to(LAB), "86de974507a39ec720712337aaffb8d48e35f6c815e22190ee3cf453d1a2030b")
    # Share four sanitized profiles, retaining their source hashes separately.
    profiles = json.loads((ROOT / "capture/PROFILES_REDACTED.json").read_text())
    for name, row in profiles.items():
        p = STAGE / "review/profiles" / (name + ".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    # The public baseline remains untouched; include reviewed helpers plus small exact source diffs.
    for rel in ["scripts", "deployment", "accepted", "profiles", "docs"]:
        tree(PUBLIC / rel, Path("public-baseline") / rel)
    for p in PUBLIC.iterdir():
        if p.is_file() and p.suffix in {".md", ".json", ".txt"} and p.name not in EXCLUDE_NAMES:
            add(p, Path("public-baseline") / p.name)
        elif p.is_file() and p.name in {"LICENSE", "NOTICE"}:
            add(p, Path("public-baseline") / p.name)
    provenance = json.loads((ROOT / "incoming/xeon-moe-peripheral-review/SOURCE_PROVENANCE.json").read_text())
    for row in provenance["source_files"]:
        add(PUBLIC / row["path"], Path("public-baseline") / row["path"], row["sha256"])
    links = []
    for row in manifest:
        if row["file"].startswith("snapshot/llm-lab/deepseek41-bringup-20261101/source/"):
            rel = Path(row["file"]).relative_to("snapshot/llm-lab/deepseek41-bringup-20261101/source")
            base = PUBLIC / "vendor/llama.cpp" / rel
            if base.is_file():
                old = base.read_bytes()
                if sha(old) != row["source_sha256"]:
                    import difflib
                    new = (STAGE / row["file"]).read_bytes()
                    try:
                        diff = "".join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True), fromfile="public-vendor/" + str(rel), tofile="current-source/" + str(rel)))
                        dst = STAGE / "review/public-vendor-diff" / (str(rel) + ".patch")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_text(diff)
                        links.append(str(rel))
                    except UnicodeDecodeError:
                        links.append(str(rel) + " (binary; current copy only)")
    meta = {"date_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "public_checkout_commit": subprocess.check_output(["git", "-C", str(PUBLIC), "rev-parse", "HEAD"], text=True).strip(),
            "public_checkout_status": subprocess.check_output(["git", "-C", str(PUBLIC), "status", "--short"], text=True),
            "engine_source_commit": subprocess.check_output(["git", "-C", str(LAB / "deepseek41-bringup-20261101/source"), "rev-parse", "HEAD"], text=True).strip(),
            "engine_source_status": subprocess.check_output(["git", "-C", str(LAB / "deepseek41-bringup-20261101/source"), "status", "--short"], text=True),
            "reviewed_external_commit": provenance["repository_commit"], "changed_current_source_vs_public_vendor": links,
            "omitted_large_text": omitted, "scope": "Review snapshot, not a portable turnkey distribution; no GGUF models, toolkit, credentials or real conversation/server logs. Source data/model fixtures and .git excluded."}
    (STAGE / "review/PACKAGE_SCOPE.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    (STAGE / "review/SOURCE_COPY_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    inventory = []
    suspicious = []
    for p in sorted(STAGE.rglob("*")):
        if p.is_file():
            raw = p.read_bytes()
            inventory.append({"path": str(p.relative_to(STAGE)), "size_bytes": len(raw), "sha256": sha(raw)})
            if p.suffix in TEXT:
                # Do not print matches; reject actual credential-shaped values for manual review.
                if re.search(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}", raw):
                    suspicious.append(str(p.relative_to(STAGE)))
    if suspicious:
        raise RuntimeError("Credential-shaped text requires review: " + repr(suspicious))
    (STAGE / "CONTENTS_SHA256.json").write_text(json.dumps({"scope": "Every archive member except this manifest; hashes are integrity checks, not signatures.", "files": inventory}, indent=2) + "\n")
    archive = ROOT / "xeon-moe-current-review-20261116.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, str(Path(STAGE.name) / p.relative_to(STAGE)))
    if archive.stat().st_size > 512 * 1024**2:
        raise RuntimeError("Archive exceeds 512 MiB")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for row in inventory:
            data = z.read(STAGE.name + "/" + row["path"])
            if len(data) != row["size_bytes"] or sha(data) != row["sha256"]:
                raise RuntimeError("ZIP manifest mismatch: " + row["path"])
    receipt = {"archive": str(archive), "sha256": sha(archive.read_bytes()), "size_bytes": archive.stat().st_size,
               "files": len(inventory) + 1, "uncompressed_bytes": sum(x["size_bytes"] for x in inventory), "ZIP_and_manifest_verified": True}
    (ROOT / "PACKAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
