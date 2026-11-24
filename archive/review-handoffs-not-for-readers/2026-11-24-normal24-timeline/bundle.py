"""Package the public measurement evidence. Identical full-vocabulary captures share one blob."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
LAB = ROOT.parent
CHECKOUT = LAB.parent / "xeon-moe-review-20261116"
DELIVERY = ROOT / "delivery"
NAME = "xeon-moe-normal24-timeline-20261124"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")


def select():
    selected = {}

    def add(path, name=None):
        assert path.is_file() and not path.is_symlink(), path
        name = str(path.relative_to(ROOT)) if name is None else name
        assert name not in selected and not Path(name).is_absolute() and ".." not in Path(name).parts
        selected[name] = path

    for name in (
        "SUMMARY_JA.txt", "SOURCES_JA.txt", "DESIGN.json", "BEFORE.json", "FINAL_API.json", "FINAL_STATE.json",
        "TIMELINE.json", "OPERATOR.json", "CPU_SAMPLES.json", "MEMORY_REFERENCE.json", "WHOLE_MODEL.json",
        "PREFILL.json", "PREFILL_CANDIDATE.json", "PREFIX_QUALITY.json", "CHECKS.json",
        "OPERATOR_STATE.json", "STATE.json", "VALIDATION_STATE.json", "PROGRESS.json"
    ):
        add(ROOT / name)
    for pattern in ("*.py", "PREFLIGHT_*.json", "WINDOW_*.json", "RESTORED_*.json"):
        for path in sorted(ROOT.glob(pattern)):
            add(path)
    directories = ("driver", "operator", "whole", "prefix_quality", "prefill", "reference", "recovery", "tools", "views")
    file_types = {".json", ".i32", ".f32", ".py", ".h", ".cpp", ".inc", ".plan"}
    for folder in directories:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in file_types and "__pycache__" not in path.parts:
                add(path)

    fixed_json = {"SPEC.json", "RUN.json", "VERIFIED.json", "PROVIDER_EXPECTED.json", "LAUNCH.json", "IMC.json",
                  "PC_DECODE.json", "PC_RECORDS.json", "trace169-summary.json"}
    native_names = {"SPEC.json", "READY.json", "STOP.json", "requests.jsonl", "IMC.json",
                    "guard-samples.jsonl", "host-counters.jsonl"}
    for run in sorted((ROOT / "runs").iterdir()):
        native = run.name.startswith("pp-")
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            if native:
                if path.parent == run and path.name in native_names:
                    add(path)
            elif path.parent != run:
                if run.name == "operator" and path.parent.name.startswith("period-") and path.suffix in (".json", ".f32"):
                    add(path)
            elif path.name in fixed_json or path.name in {"guard-samples.jsonl", "host-counters.jsonl", "server.stderr.log", "server.stdout.log"}:
                add(path)
            elif path.suffix in (".f32", ".tsv") or re.fullmatch(r"pc-[01]-(armed|ready|done)\.json", path.name):
                add(path)

    adoption = LAB / "adopt24-20261124"
    prior = LAB / "postload-measurement-20261124"
    observer = LAB / "169a-exploration-20261119/astra62"
    external = {
        "tools/trace169.cpp": observer / "trace169.cpp",
        "tools/cupti-BUILD.json": observer / "observer/BUILD.json",
        "tools/observer-build.py": observer / "observer/build.py",
        "tools/read_bandwidth.c": LAB / "read_bandwidth.c",
        "tools/check_live_identity.py": LAB / "results-review-20261124/tools/check_live_identity.py",
        "reference/runtime.json": adoption / "current/runtime.json",
        "reference/previous-adoption.txt": adoption / "SUMMARY_JA.txt",
        "reference/previous-fixed-api.txt": prior / "SUMMARY_JA.txt",
        "reference/expected-normal.json": LAB / "results-review-20261124/reference/EXPECTED_NORMAL_24_24.json",
        "reference/ggml-backend.cpp": LAB / "gpu-stream-20261103/ggml-backend.cpp",
        "reference/ggml-cuda.cu": LAB / "exploration-20261112/cuda-sort/ggml-cuda.cu",
        "licenses/project-LICENSE": CHECKOUT / "LICENSE",
        "licenses/llama.cpp-LICENSE": LAB / "deepseek41-bringup-20261101/source/LICENSE",
        "licenses/ACKNOWLEDGMENTS.md": CHECKOUT / "ACKNOWLEDGMENTS.md",
        "licenses/THIRD_PARTY_NOTICES.md": CHECKOUT / "THIRD_PARTY_NOTICES.md"
    }
    for name, path in external.items():
        add(path, name)
    assert digest(external["tools/read_bandwidth.c"]) == json.loads((ROOT / "MEMORY_REFERENCE.json").read_bytes())["source_sha256"]
    return selected


def arrange(tree):
    captures = {}
    for name, source in sorted(select().items()):
        if name.endswith(".f32"):
            h = digest(source)
            obj = "objects/" + h + ".f32"
            captures[name] = {"sha256": h, "bytes": source.stat().st_size, "object": obj}
            target = tree / obj
        else:
            target = tree / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    save(tree / "F32_INDEX.json", {"captures": captures})
    raw = {"observer_elf": LAB / "169a-exploration-20261119/astra62/observer/libtrace169.so",
           "bandwidth_elf": LAB / "read-bandwidth", "libgomp_debug": ROOT / "private/libgomp.debug"}
    for stage in ("driver", "operator", "whole", "prefix_quality"):
        raw[stage + "_driver"] = ROOT / stage / "driver"
    for n in (0, 1):
        raw[f"perf_script_{n}"] = ROOT / f"runs/observe/pc-{n}.local.txt"
    sums = {key: {"sha256": digest(path), "bytes": path.stat().st_size} for key, path in raw.items()}
    native = json.loads((tree / "reference/perf_native_hashes.json").read_bytes())
    for info in json.loads((tree / "runs/observe/PC_DECODE.json").read_bytes()):
        n = info["case"]
        p = ROOT / f"runs/observe/pc-{n}.local.data"
        assert info["input_sha256"] == native[p.name]
        sums[f"perf_data_{n}"] = {"sha256": native[p.name], "bytes": p.stat().st_size}
    gomp = json.loads((tree / "CPU_SAMPLES.json").read_bytes())
    assert sums["libgomp_debug"]["sha256"] == gomp["debug_sha256"]
    save(tree / "SOURCE_HASHES.json", {"native_local_files": sums, "debug_source": gomp["debug_source"]})


def make_manifest(tree):
    patterns = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{28,}|github_pat_[A-Za-z0-9_]{45,}|sk-[A-Za-z0-9]{30,}|Authorization: Bearer [A-Za-z0-9_./+\-]{12,}", re.I)
    secret = (LAB / "api-server-20261030/api-key.txt").read_bytes().strip()
    assert len(secret) >= 12
    files = []
    for p in sorted(tree.rglob("*")):
        if not p.is_file():
            continue
        name = str(p.relative_to(tree))
        assert p.name not in ("api-key.txt", "server-config.json", "health.json")
        data = p.read_bytes()
        assert not p.is_symlink() and not data.startswith(b"\x7fELF") and len(data) < 100 * 1024**2
        assert secret not in data and not patterns.search(data), name
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    save(tree / "MANIFEST_SHA256.json", {"files": files})
    return files


def main():
    DELIVERY.mkdir(exist_ok=True)
    target = DELIVERY / NAME
    zp = DELIVERY / (NAME + ".zip")
    assert not target.exists() and not zp.exists()
    with tempfile.TemporaryDirectory(dir=DELIVERY) as work:
        tree = Path(work) / NAME
        tree.mkdir()
        arrange(tree)
        entries = make_manifest(tree)
        archive = Path(work) / zp.name
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for path in sorted(tree.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(tree))
        assert archive.stat().st_size < 512_000_000
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(entries) + 1 and len(set(z.namelist())) == len(z.namelist())
            z.extractall(tmp)
            proof = subprocess.check_output([sys.executable, "-B", str(Path(tmp) / "replay.py"), tmp],
                                            text=True, timeout=350).splitlines()
        result = {"filename": zp.name, "bytes": archive.stat().st_size, "sha256": digest(archive),
                  "files": len(entries) + 1, "local_extracted_replay": proof}
        tree.rename(target)
        archive.rename(zp)
    zp.with_suffix(".zip.sha256").write_text(result["sha256"] + "  " + zp.name + "\n")
    save(DELIVERY / "PACKAGE.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
