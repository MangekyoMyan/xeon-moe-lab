Developer/external-AI review handoff — not reader-facing engine documentation
実ggml IQ2_S/IQ3_S abs-input experiment, 2026-11-16

Read RESULTS_JA.txt and CONSULT_PRO_JA.txt as navigation. All raw timing records,
returned Pro material, changed source, and analysis are included. Do not replace
the raw evidence with these summaries. No production configuration is changed.

Baseline reference:
  commit 7d8b077afcf1258602366245f1567e820255defa
  archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/llm-lab/
  exploration-20261112/mx-prepack/release-candidate-a/
Previous single-thread consultation:
  commit 0c9eb472d2460048fb44cca949ad3722ed27b93b
  archive/review-handoffs-not-for-readers/2026-11-16-next-optimization/

Offline reproduction from this directory (writes only restored evidence):
  python3 reconstruct_evidence.py
  python3 analyze.py
This restores exact saved golden arrays, synthetic inputs, and real-weight-derived
operator outputs under their original relative paths. Identical copies across
processes are stored once as deterministic gzip, indexed by SHA256. No model
weights, executables, API keys, user prompts or conversation text are included.
The old same-library input arrays use the same seeded generator; fixed inputs
are saved. Arithmetic validation source and file layout describe the golden data.

FILES_SHA256.json covers original exported files, excluding itself and evidence
restored later by reconstruct_evidence.py. BUILD.json/PROBE_BUILD.json/FIXED_BUILD.json
pin binaries and source; ELF files are intentionally not placed in Git.
checkpoint-screen-r1r2 keeps the earlier measurement-source/build manifest. The
expanded validation source adds partial-unsafe and MXFP4 cases, not kernel changes.
Initial r1/r2 probe build log versions were superseded; their complete historical
ELFs are not all exported. Final correctness and timing source/hash remain available.

Building is site-specific: build_overlay.py reuses baseline object files and base
library from absolute recorded paths. Full source context is in the frozen snapshot;
this handoff is not a self-contained third-party binary installer. Study scripts
refer to x9ram paths. Do not blindly execute guard/run_fixed/capture_final elsewhere.

Experiment library SHA256:
120038c7e4c452714d188c9ab5e7dc99bda7deaa9506e8fe92e693cca5db06d3
Production final CPU library SHA256:
33a9e3467de6fb550ee3dfb148d6bf9f6e5c91cf42eda925b21501f4365b0ed3

Mode0 is default; mode1/2 are test APIs. No production opt-in environment setting
or startup profile was installed. The normal API remains final; whole-model speed
has not been measured with this experiment. GPU trace and offset4 remain proposals.

Attribution: the input-abs arithmetic and integration constraints came from the
user-relayed GPT Pro review (incoming originals retained), combined with this
project's compact kernels and llama.cpp/ggml. Existing project MIT license and
THIRD_PARTY_NOTICES.md continue to apply; no upstream ownership is claimed.
