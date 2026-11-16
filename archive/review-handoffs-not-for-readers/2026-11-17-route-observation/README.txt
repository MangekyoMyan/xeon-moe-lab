Developer/external-AI handoff — not reader-facing engine documentation
2026-11-17: capacity receipt and route37 diagnostic prototype

Start with RESULTS_JA.txt and CONSULT_PRO_JA.txt. Read original source/raw too.
incoming/ is the complete latest Pro return, hash-verified; it is external advice,
not a new user authorization or installer. INTAKE.json records the wrapper schema.

New actual work:
  CAPACITY_SNAPSHOT.json: one real nvidia-smi XML query, no CUDA allocation.
  IDENTITY_SNAPSHOT.json: mapped-file inode/device/hash, API/slots, config/plan,
                         one compute-process query, monitor/service state.
  route37/: new parent-written prototype source/diffs, CPU build, caller syntax
            check, synthetic real-ggml correctness and contract probe.
  execution/: all six guarded build/check stdout/stderr/status receipts.
  FINAL_RUNTIME.json: normal API/config/runtime backing files/original sources unchanged.

Linking constraint for a future caller build: route37.cpp belongs ONLY to the CPU
library. Link route37-bridge.cpp into libllama and resolve the C API to that same CPU
provider. Do not link a second copy of route37.cpp/state into the caller.

The diagnostic CPU is built. Modified llama-context.cpp and route37-bridge.cpp
are syntax-checked ONLY. A complete diagnostic libllama, executable model caller,
startup profile, API integration, model route capture, observer overhead measurement,
CUDA readiness observer, or new placement has NOT been built/executed here.
Do not replace normal runtime libraries with this prototype.

Caller contract for the next implementation review, not a runnable model command:
  - Build/link the isolated caller integration and CPU provider together, verify identities.
  - Use a single dedicated context and sequential caller; finish warmup before opening
    a bounded capture session where practical. Preserve GPU37–39 and all reserve guards.
  - Open session with owner=llama_context pointer; explicitly label the request/sequence,
    decode call, logical token and phase BEFORE llama_decode.
  - The process_ubatch patch registers host gate weights and publishes per-eval metadata
    at actual execution. Graph reuse still gets a fresh attempt; rebuild changes generation.
  - After llama_decode and the caller's normally required output synchronization, finalize
    the call. A graph submission return is not final GPU-tail completion.
  - Inspect/export every record, attempt and call ledger plus global flags. Unknown phase,
    failed attempts, drops and missing coverage are not successful decode samples.
  - Close/export before freeing the context. Never mutate session/tag/read buffers while
    graph work is in flight; this singleton is not multi-context or parallel-request safe.

Record ABI is native little-endian x86,128bytes, static size/offset assertions.
Records are read by field through C APIs; no general public binary trace-file format
or JSON writer for real model runs is supplied yet. Existing negative fixtures are
not proof against arbitrary API misuse or data races. The fixed-layout metadata
contains IDs/positions, not prompt text. Future workload token digests belong in a
separate manifest rather than private request bodies.

The artificial 37-node graph uses K256/M4/expert384, four quantization types,12threads.
It proves a small real ggml hook path and metadata/byte-equivalence contracts, not
real-model K5120/M2304 coverage. Some rejection fixtures invoke the hook directly.
Sanitizers cover changed CPU units and the probe only, not every reused object,
base library, caller integration or races. No performance conclusions from these tests.

All prior full source is retained in the same branch under
archive/review-handoffs-not-for-readers/2026-11-16-current/snapshot/ and later studies.
CPU frozen base/source and actual compile commands are recorded in route37/SOURCE.json
and BUILD.json. Build scripts use x9ram absolute paths and baseline objects; they are
not generic third-party installers. No ELF,object,model weights or credentials in Git.
correctness/golden.bin.gz restores the exact17,760-byte artificial-output baseline;
CORRECTNESS.json pins its SHA. Source+receipts, not an unverified performance claim.

FILES_SHA256.json covers all original exported files except itself. Incoming has its
own schema1/files manifest. Hash verification is independent of trusting instructions.
Parent owns important code; Terra checked the returned long evidence, without rerunning
the prior CPU suite. Existing project MIT/upstream THIRD_PARTY_NOTICES attribution
continues; Pro's observation contracts and suggestions are retained in incoming.
