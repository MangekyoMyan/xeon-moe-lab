#!/usr/bin/env python3
"""Switch the normal launcher after canonical correctness and speed checks pass."""
import datetime
import hashlib
import json
from pathlib import Path

lab = Path(__file__).resolve().parent
results = lab / "results"
def read(path):
    return json.loads(path.read_text())

source_record = results / "deep-review-source-adoption.json"
adoption = read(source_record)
if adoption["launcher_switched"]:
    raise SystemExit("The launcher was already switched")
if not read(results / "deep-review-functional.json")["all_passed"] or not read(results / "deep-review-rebuild-check/validation.json")["all_passed"]:
    raise SystemExit("Canonical correctness checks must pass first")
comparisons = [(p, read(p)) for p in results.glob("cpu-abba-20261019T*/comparison.json")]
matched = [(p, r) for p, r in comparisons if r["experiment_note"].startswith("Current launcher build versus canonical combined rebuild")]
if {r["order"] for _, r in matched} != {"ABBA", "BAAB"}:
    raise SystemExit("Canonical A/B comparisons in both orders are required")
for _, comparison in matched:
    if comparison.get("candidate_gain_percent", -100) <= 0 or not comparison.get("same_boot") or not comparison.get("warm_cache_observed"):
        raise SystemExit("Review canonical performance before switching")
for current, backup, staged in (("run_deepseek_flash.sh", "run_deepseek_flash_before_deep_review.sh", "run_deepseek_flash_deep_staged.sh"),
                                 ("rebuild_deepseek.sh", "rebuild_deepseek_before_deep_review.sh", "rebuild_deepseek_deep_staged.sh")):
    if (lab / current).read_bytes() != (lab / backup).read_bytes():
        raise SystemExit("Current configuration changed; inspect " + current)
    if not (lab / staged).is_file():
        raise SystemExit("Missing staged configuration")
for current, staged in (("run_deepseek_flash.sh", "run_deepseek_flash_deep_staged.sh"),
                         ("rebuild_deepseek.sh", "rebuild_deepseek_deep_staged.sh")):
    temporary = lab / (current + ".deep-new")
    temporary.write_bytes((lab / staged).read_bytes())
    temporary.chmod(0o755)
    temporary.replace(lab / current)
adoption.update({"launcher_switched": True, "switched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "canonical_performance_comparisons": [str(p) for p, _ in matched],
                 "launcher_sha256": hashlib.sha256((lab / "run_deepseek_flash.sh").read_bytes()).hexdigest(),
                 "numa_replica_enabled_by_default": False})
source_record.write_text(json.dumps(adoption, indent=2) + "\n")
preferences_path = lab / "experiment-preferences.json"
preferences = read(preferences_path)
preferences["previous_ssse3_work_summary"] = preferences["current_work"]
preferences["current_work"] = ("Adopted private build-avx-deep: prior SSSE3 IQ unpack plus exact paired IQ2 MoE gate/up and four-row AVX Q8 dot for K8192. "
    "OpenMP ON, BLAS OFF, generation12/prefill24, mmap/interleave, batch512. Canonical decode rounds +3.23%/+2.61%; isolated candidate +4.61%/+5.13%, prefill +2.14%. "
    "The fixed short Japanese response ABBA changed by only +0.13%, so practical short-answer speed is effectively unchanged. "
    "Full-model fixed-sequence logits and two Japanese answers agree with baseline. "
    "NUMA weight replication remains optional (GGML_CPU_NUMA_REPLICA=1): about 6.06 GiB extra RSS for +0.85% on combined decode. Old launcher/build retained.")
preferences["deep_review_results"] = str(results / "deep-review-final/REPORT.json")
preferences["deep_review_archive"] = str(lab.parent / "xeon-deep-optimization-results-20261019.zip")
preferences["active_build"] = str(lab / "build-avx-deep")
preferences_path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(adoption, indent=2))
