Developer/external-AI handoff — not reader-facing engine documentation
IQ2 mixed-cache study on Xeon, 2026-11-17

RESULTS_JA.txt: outcome, limits, unchanged runtime, why the adoption screen failed.
CONSULT_PRO_JA.txt: next decision request; summaries are navigation, not substitutes for raw.
POLICY_AND_PLAN_JA.txt: user steering vs external proposals, and prospective scope.
incoming/: the full Pro return, byte-preserved and hash-verified; its synthetic checks
            are NOT this study's Xeon performance measurements.
mixed-run/process-0..7/: actual launch receipts, raw timestamps, maps, stdout/stderr.
outer-guard/: actual outer guard receipts for build and all eight processes.
MIXED_ANALYSIS.json / SCREEN_DECISION.json: complete offline analysis and prospective gate.
MIXED_RAW_REVIEW_TERRA.txt: independent raw-total/time-window reaggregation.
build/BUILD.json: actual link command/source/harness SHA; no CPU library rebuild.

The incoming mixed_probe.cpp, launcher and analyzer were used without source edits.
Their guard does not replace the existing daemon/API activity guard. All eight
processes used the same harness/base/inputs/plan; F/E CPU hashes stayed fixed.
The engine/API remain final. No new production option was installed.

Restore and reanalyze locally from this directory (standard Python only):
  python3 reconstruct_evidence.py
  python3 -B incoming/xeon_moe_abs_next_2026-11-16/next_experiment/analyze_mixed_run.py \
    --run-root mixed-run --output REPLAY_ANALYSIS.json
The output contains a machine-local run_root. All other analysis data should match
MIXED_ANALYSIS.json. The analyzer refuses to overwrite an existing output.
The saved 24 mode0 expected files are deduplicated as exact gzip blobs; INDEX.json
lists destinations and original hashes. Candidate timed-chunk final outputs were
compared in memory by the harness; all 32 intermediate outputs were not saved.
inputs.bin in incoming is a fixed synthetic fixture, not a model weight file.

Full engine/source context and prior results:
  commit31b1292b2767618a28ef6e3f1c9c4450b1fc21be
  archive/review-handoffs-not-for-readers/2026-11-16-abs-operator/
  archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/
This branch retains both. Build/run scripts record x9ram absolute paths; they are
not general installers and should not be blindly launched on another environment.
No model weights, ELF, API keys or inference-request conversation payloads are exported.
The policy note includes the user's explicit operating preferences.

FILES_SHA256.json covers original export files except itself and later restored
expected arrays. Every timed observation and all eight process outcomes remain.
No trimming, retry-until-pass, new performance threshold, or whole-model claim.

Attribution: mixed protocol, harness, launcher, analyzer and reviewed constraints
were supplied through the user-relayed GPT Pro review. Existing project MIT and
THIRD_PARTY_NOTICES.md apply to project source; upstream llama.cpp/ggml attribution
is retained. Review checks from the other machine are labeled separately.
