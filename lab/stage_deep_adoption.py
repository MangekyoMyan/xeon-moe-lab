#!/usr/bin/env python3
"""Copy the verified private changes and prepare a separate rebuild and launcher."""
import hashlib
import json
from pathlib import Path
import shutil

lab = Path(__file__).resolve().parent
candidate = lab / "review-deep-combined-20261019"
private = lab / "llama-private"
manifest = json.loads((candidate / "prepare.json").read_text())
record = lab / "results/deep-review-source-adoption.json"
if record.exists():
    raise SystemExit("Preserve the existing adoption record")
for relative, expected in manifest["files"].items():
    old, new = private / relative, candidate / "source" / relative
    before = hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None
    after = hashlib.sha256(new.read_bytes()).hexdigest()
    if before != expected["before"] or after != expected["after"]:
        raise SystemExit("Source changed; review before adoption: " + relative)
for source, backup in (("run_deepseek_flash.sh", "run_deepseek_flash_before_deep_review.sh"),
                       ("rebuild_deepseek.sh", "rebuild_deepseek_before_deep_review.sh")):
    if (lab / backup).exists():
        raise SystemExit("Preserve existing backup: " + backup)
for relative in manifest["files"]:
    shutil.copy2(candidate / "source" / relative, private / relative)
for source, backup in (("run_deepseek_flash.sh", "run_deepseek_flash_before_deep_review.sh"),
                       ("rebuild_deepseek.sh", "rebuild_deepseek_before_deep_review.sh")):
    shutil.copy2(lab / source, lab / backup)
launcher = (lab / "run_deepseek_flash.sh").read_text().replace("build-avx-unpack", "build-avx-deep")
launcher = launcher.replace("# Measured CPU profile: OpenMP on, BLAS off, SSSE3 IQ unpack.",
    "# Measured CPU profile: OpenMP on, BLAS off, SSSE3 IQ unpack, paired MoE and Q8 rows4.\n"
    "# Optional: GGML_CPU_NUMA_REPLICA=1 uses about 6 GiB extra RSS for a small generation gain.")
staged_launcher = lab / "run_deepseek_flash_deep_staged.sh"
staged_launcher.write_text(launcher)
staged_launcher.chmod(0o755)
staged_rebuild = lab / "rebuild_deepseek_deep_staged.sh"
staged_rebuild.write_text("""#!/usr/bin/env bash
set -euo pipefail

# Private source: upstream 436f6f89e plus the locally verified CPU changes.
cmake -S /home/codex/projects/llm-lab/llama-private \\
  -B /home/codex/projects/llm-lab/build-avx-deep \\
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=ON \\
  -DGGML_OPENMP=ON -DGGML_BLAS=OFF \\
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF \\
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=OFF \\
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build /home/codex/projects/llm-lab/build-avx-deep \\
  --target llama-bench llama-completion test-backend-ops test-quantize-fns -j 8
""")
staged_rebuild.chmod(0o755)
record.write_text(json.dumps({"source": str(private), "files": manifest["files"],
    "launcher_switched": False, "build": str(lab / "build-avx-deep"),
    "rebuild": str(staged_rebuild), "staged_launcher": str(staged_launcher)}, indent=2) + "\n")
print(record)
