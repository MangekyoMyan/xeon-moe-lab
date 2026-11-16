Developer/external-AI handoff — not reader-facing engine documentation
2026-11-17: full route37 libllama link, scheduler integration and user-run driver

Read RESULTS_JA.txt, USER_RUN_JA.txt and CONSULT_PRO_JA.txt. The original incoming
review, source and complete local logs are retained; summaries are navigation.

Runtime status: ordinary final API remains unchanged. No model was loaded or API
restarted for this study. Diagnostic libllama and driver are now linked; a manual
replacement capture is prepared but NOT yet validated on the actual model.

Important evidence:
  LINK_BUILD.json: pinned baseline relink is byte-identical, reused object SHA,
                  diagnostic link arguments and source/Unity correspondence.
  provenance/: link map, dynamic/defined/undefined symbols, actual model sources,
               Unity include and dependency. No ELF/object is included in Git.
  PROVIDERS_FINAL.json + PROVIDER_VERIFICATION_FINAL.json:16 pointers, paths,
               inode/device/backing hashes before backend initialization.
  probe-results-v2/: final real CPU-only scheduler3-graph/111-record fixture and8 caller
                  callback/direct-hook cases. No actual model/GPU execution.
  execution/: all11 guarded steps, including the resolved initial check/build errors.
  CLIENT_BUILD.json: final executable and source hashes.
  LAUNCH_PLAN.json: selected normal-profile arguments/environment, no API secret.
  LAUNCH_REFUSAL_TEST.json: mocked active-service refusal, Popen forbidden.
  FINAL_RUNTIME.json: ordinary API/config/files still unchanged and health ok.

route37.cpp/state is defined ONLY in the already built diagnostic CPU library from
the previous observation study. The header comment update leaves non-comment tokens
unchanged. The new libllama contains the bridge and references that CPU's C API.
No second state is linked into driver/libllama. The caller patch additionally closes
a CPU attempt as FAILED before rethrowing a graph-compute C++ exception.

The current actual-model graph passes each layer's original ffn_gate_exps to generic
build_moe_ffn (actual deepseek41.cpp around the layer loop is exported). This source
reasoning and identical baseline object relink are not a substitute for the manual
four-token endpoint coverage check. Whole source recompilation was not performed.

Driver uses public decode results and normal synchronized logits on success; it
drains submitted work before finalization on failure. Full JSON is exported by field,
not a native struct memory dump. A failed finite-output check is retained separately
and fails the run even if the collector's call-completion count is nonzero.
The fake-callback tests are not actual CUDA fault injection or a new sanitizer pass.

User startup: read USER_RUN_JA.txt. Default4 calls; explicit --continue64 advances
only after4-call coverage passes, then captures64 more in the same context.
The fixed input is teacher-forced artificial text with a64-token prefix, not a
natural response or representative traffic. Observer overhead and whole-model
output equivalence have not been measured. CPU abs and positive hot remain on hold.

The launcher reuses the current production supervisor with its thermal/ECC/replay/
fan checks and experiment lock, redirects its files to a fresh diagnostic directory,
requires64GiB RAM reserve and adds a2hour bound. It never stops/restores the API or
changes hardware settings. --prepare only reads files and writes a plan.
The actual long-load path, vision warmup and capture still require the user run.

Local build/run scripts preserve x9ram paths and depend on baseline objects/library
files outside this directory. This is source/evidence, not a portable binary installer.
Previous source studies are in the same branch at 688683c/ff616809 and the full current
snapshot under archive/review-handoffs-not-for-readers/2026-11-16-current/.

FILES_SHA256.json covers this export except itself. Checkpoints preserve earlier
source/build receipts; their hashes intentionally describe earlier artifacts.
Expected failure logs must not be counted as successful test receipts or omitted.
Current project MIT and THIRD_PARTY_NOTICES remain; Pro's API comment and integration
recommendations are preserved in incoming, and llama.cpp/common/mtmd attribution
continues. Model weights, build ELF/object, credentials and real user prompt payloads
are not exported. The only .bin in incoming is the artificial17,760B golden fixture.
