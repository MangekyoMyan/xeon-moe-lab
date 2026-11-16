#!/usr/bin/env python3
"""Read-only review of the frozen repo; all mutations stay in a TemporaryDirectory.

No model, service, GPU, or network operation is performed. CMake only configures
CPU targets; its small compiler-feature probes may compile/run trivial programs.
The UI script's two network DOWNLOAD calls are replaced in an isolated copy by
a deterministic failure stub. This is a branch-reachability test, not a download.
Requires the frozen commit with no tracked working-tree changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

EXPECTED_COMMIT = "4ff68eba2d8c480692511dc6d219d27a6a102866"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--cmake", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-cmake-configure", action="store_true")
    a = ap.parse_args()
    repo, cmake, output = a.repo.resolve(), a.cmake.resolve(), a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        ap.error("--output must be a new file")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if rev != EXPECTED_COMMIT:
        ap.error(f"Expected {EXPECTED_COMMIT}, found {rev}")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True)
    if tracked_status:
        ap.error("Refusing dirty tracked working tree; use a clean checkout of the frozen commit")
    before_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    env = os.environ.copy()
    # Neither credentials nor downloads are needed for this reproduction.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CC", "CXX", "CUDACXX"):
        env.pop(name, None)
    findings: dict = {
        "repo_commit": rev,
        "tracked_working_tree_clean_at_start": True,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "scope": "temporary copies, no model/service/GPU/network; configure only",
        "source_scripts_sha256": {
            p: sha(repo / p) for p in (
                "scripts/prepare_source.py", "scripts/build_plan.py", "scripts/verify_runtime.py",
                "vendor/llama.cpp/scripts/ui-assets.cmake",
            )
        },
        "commands": [],
    }

    def run(cmd: list[str], label: str, cwd: Path | None = None) -> dict:
        started = time.monotonic()
        result = subprocess.run(cmd, cwd=cwd or repo, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=180)
        record = {"label": label, "argv": cmd, "returncode": result.returncode,
                  "elapsed_s": time.monotonic() - started,
                  "stdout": result.stdout, "stderr": result.stderr}
        findings["commands"].append(record)
        return record

    with tempfile.TemporaryDirectory(prefix="xeon-build-review-") as td:
        temp = Path(td)
        source = temp / "accepted-source"
        prep = [sys.executable, str(repo / "scripts/prepare_source.py"),
                "--destination", str(source)]
        assert run(prep, "prepare_source_fresh")["returncode"] == 0
        assert run(prep + ["--verify"], "verify_source_fresh")["returncode"] == 0

        # Verify both a changed upstream source and a missing upstream header.
        quant = source / "ggml/src/ggml-quants.c"
        original = quant.read_bytes()
        quant.write_bytes(original + b"\n#error AUDIT_REPRO_UNCHECKED_UPSTREAM\n")
        changed = sha(quant)
        altered = run(prep + ["--verify"], "verify_changed_nonoverlay_source")
        quant.write_bytes(original)
        header = source / "include/llama.h"
        original_header = header.read_bytes()
        header.unlink()
        absent = run(prep + ["--verify"], "verify_missing_nonoverlay_header")
        header.write_bytes(original_header)
        overlay = source / "xeon-local/iq3-cache.h"
        old_overlay = overlay.read_bytes()
        overlay.write_bytes(old_overlay + b"\n/* audit altered overlay */\n")
        overlay_check = run(prep + ["--verify"], "verify_changed_overlay_positive_control")
        overlay.write_bytes(old_overlay)
        findings["prepared_tree_verification"] = {
            "changed_file": "ggml/src/ggml-quants.c",
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "changed_sha256": changed,
            "changed_nonoverlay_accepted": altered["returncode"] == 0,
            "missing_header_accepted": absent["returncode"] == 0,
            "changed_overlay_rejected": overlay_check["returncode"] != 0,
            "note": "Only a temporary prepared tree was modified. No changed source was compiled.",
        }

        plan = run([sys.executable, str(repo / "scripts/build_plan.py"),
                    "--cuda-root", str(temp / "nonexistent-cuda")], "print_build_plan")
        assert plan["returncode"] == 0
        argv = json.loads(plan["stdout"])[0]
        options = {item[2:].split("=", 1)[0]: item[2:].split("=", 1)[1]
                   for item in argv if item.startswith("-D")}
        findings["plan_explicit_options"] = options

        # Keep every provisioning branch except the network primitive unchanged.
        ui_original = (repo / "vendor/llama.cpp/scripts/ui-assets.cmake").read_text()
        assert ui_original.count("file(DOWNLOAD ") == 2
        ui_stub = """function(audit_download)
    message(STATUS "AUDIT_BLOCKED_DOWNLOAD ${ARGV0}")
    set(status "1;offline audit mock" PARENT_SCOPE)
endfunction()
"""
        ui_script = temp / "ui-assets-no-network.cmake"
        ui_script.write_text(ui_stub + ui_original.replace("file(DOWNLOAD ", "audit_download("))
        ui_results = {}
        for enabled in ("ON", "OFF"):
            ui_out = temp / ("ui-" + enabled.lower())
            ui_out.mkdir()
            record = run([
                str(cmake), "-DBUILD_UI=OFF", f"-DHF_ENABLED={enabled}",
                "-DHF_BUCKET=ggml-org/llama-ui", "-DHF_VERSION=b1",
                "-DLLAMA_UI_GZIP=OFF", f"-DUI_SOURCE_DIR={source / 'tools/ui'}",
                f"-DUI_BINARY_DIR={ui_out}", f"-DLLAMA_SOURCE_DIR={source}",
                "-P", str(ui_script),
            ], f"ui_branch_hf_{enabled.lower()}")
            ui_results[enabled] = {
                "returncode": record["returncode"],
                "blocked_download_urls": [line.split("AUDIT_BLOCKED_DOWNLOAD ", 1)[1]
                                          for line in record["stdout"].splitlines()
                                          if "AUDIT_BLOCKED_DOWNLOAD " in line],
                "generated_ui_cpp": (ui_out / "ui.cpp").exists(),
            }
        findings["ui_provisioning"] = {
            "test_kind": "exact script with only file(DOWNLOAD) replaced by no-network failure stub",
            "HF_VERSION_test_value": "b1 (explicit fixture value, not a claim about the real build number)",
            "plan_sets_build_ui_off": options.get("LLAMA_BUILD_UI") == "OFF",
            "plan_sets_prebuilt_ui": options.get("LLAMA_USE_PREBUILT_UI"),
            "cases": ui_results,
        }

        if not a.skip_cmake_configure:
            # Same plan options except disable unavailable CUDA. Do not build any
            # target, so UI provisioning, server startup and kernels cannot run.
            adapted = {k: v for k, v in options.items()
                       if not k.startswith("CMAKE_CUDA_") and k != "CUDAToolkit_ROOT"}
            adapted["GGML_CUDA"] = "OFF"

            def configure(where: Path, extra: list[str], label: str) -> dict:
                return run([str(cmake), "-S", str(source), "-B", str(where),
                            *[f"-D{k}={v}" for k, v in adapted.items()], *extra], label)

            reused = temp / "build-reused"
            fresh = temp / "build-fresh"
            seed = configure(reused, ["-DGGML_SANITIZE_ADDRESS=ON", "-DGGML_AMX_TILE=ON"],
                             "configure_seed_nonbaseline_cache")
            rerun = configure(reused, [], "configure_plan_reusing_cache")
            clean = configure(fresh, [], "configure_plan_fresh_cache")

            def details(build: Path) -> dict:
                cache = {}
                if (build / "CMakeCache.txt").exists():
                    for line in (build / "CMakeCache.txt").read_text().splitlines():
                        if not line.startswith(("//", "#")) and "=" in line:
                            k, v = line.split("=", 1)
                            cache[k.split(":", 1)[0]] = v
                commands_path = build / "compile_commands.json"
                commands = json.loads(commands_path.read_text()) if commands_path.exists() else []
                entry = next((x for x in commands if x["file"].endswith("/ggml-cpu.c")), {})
                return {
                    "cache": {k: cache.get(k) for k in (
                        "GGML_SANITIZE_ADDRESS", "GGML_AMX_TILE", "LLAMA_USE_PREBUILT_UI",
                        "LLAMA_BUILD_UI", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
                    )},
                    "ggml_cpu_compile_command": entry.get("command", entry.get("arguments")),
                }
            findings["cache_reuse"] = {
                "scope": "real prepared project, configure only; CUDA OFF due unavailable toolkit",
                "seed_returncode": seed["returncode"],
                "rerun_returncode": rerun["returncode"],
                "fresh_returncode": clean["returncode"],
                "reused": details(reused), "fresh": details(fresh),
                "build_invoked": False,
            }

        # Verifier scope checks use a tiny well-formed artificial runtime, not
        # the release binaries. These distinguish byte checks from inventory.
        runtime = temp / "runtime"
        (runtime / "bin").mkdir(parents=True)
        lib = runtime / "bin/libfixture.so.0"
        lib.write_bytes(b"not an ELF; only byte-integrity verifier fixture\n")
        expected = lib.read_bytes()
        link = runtime / "bin/libfixture.so"
        link.symlink_to(lib.name)
        manifest = runtime / "RUNTIME_MANIFEST.json"
        data = {"files": [
            {"file": "bin/libfixture.so.0", "sha256": sha(lib), "bytes": lib.stat().st_size},
            {"file": "bin/libfixture.so", "symlink": lib.name},
        ]}
        manifest.write_text(json.dumps(data))
        verify = [sys.executable, str(repo / "scripts/verify_runtime.py"), str(runtime)]
        valid = run(verify, "runtime_valid_control")
        lib.write_bytes(expected + b"changed")
        corrupt = run(verify, "runtime_corrupt_listed_file_control")
        lib.write_bytes(expected)
        (runtime / "bin/libgomp.so.1").write_bytes(b"unlisted fixture, not loaded\n")
        extra = run(verify, "runtime_unlisted_file")
        manifest.write_text(json.dumps({"files": []}))
        empty = run(verify, "runtime_empty_manifest")
        findings["runtime_verifier_scope"] = {
            "valid_control_passed": valid["returncode"] == 0,
            "corrupt_listed_file_rejected": corrupt["returncode"] != 0,
            "unlisted_file_accepted": extra["returncode"] == 0,
            "empty_manifest_accepted": empty["returncode"] == 0,
            "scope": "synthetic byte/symlink fixtures; no ELF or release binary executed",
            "interpretation": "Inventory/root-of-trust limitation, not evidence that current release is corrupt",
        }

    after_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    findings["git_status_before"] = before_status
    findings["git_status_after"] = after_status
    findings["repo_status_unchanged"] = before_status == after_status
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(findings, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "repo_status_unchanged": before_status == after_status,
                      "prepared_tree": findings["prepared_tree_verification"],
                      "ui": ui_results, "runtime": findings["runtime_verifier_scope"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
