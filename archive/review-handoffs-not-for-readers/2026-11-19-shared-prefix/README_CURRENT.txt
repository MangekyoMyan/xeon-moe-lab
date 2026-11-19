2026-11-19 shared-expert prefix exploration

Start with RESULTS_JA.txt and CONSULT_PRO_JA.txt for final outcome and next questions. This is a review handoff and development record, not a general-purpose supported engine release.

Executed source
  main/driver.cpp: ordinary-provider baseline and original-graph CUPTI diagnostic.
  astra62/trace169.cpp: observer-only public-dispatch interposer and CUPTI collector.
  prefix-engine/ggml-backend.cpp + shared_prefix.inc: parent-reviewed integrated trial base.
  prefix-engine/BUILD.json: exact frozen2e13 control reproduction and actualcandidate build.
  trial/driver.cpp: fixed-input caller with qualified37-layer execution-count checks.
  run_main.py / run_trial.py / api_window.py / verify_restore.py: host supervision and ordinaryAPI restoration. Do not run these as offline analysis commands.

Historical proposals/corrections
  astra62/prefix-candidate is the submitted design, including its preserved initial teardown defect. It is not the executed candidate. PROPOSAL_STATUS_JA.txt explains the correction. Parent source and build are in prefix-engine.
  checkpoint-verifier-before-stdlib preserves a verifier-environment failure after a successful ordinary model run. Baseline was verified without rerunning it.
  prefix-engine/compile-fix-checkpoint preserves a compile-only C-linkage macro error; control had passed, candidate was not executed. Corrected source and retryreceipt are current.
  Initial draft sync-on-scheduler-free could dereference a backend already destroyed by the owning llama_context. Independent source review caught this before candidate build/run. Actual code relies on originalowner quiescence and live-backend error drains.

Evidence
  main/runs: original baseline and observer, all65rows percase.
  trial/runs: uninstrumentedcandidatequality, candidateobserver, and fresh A/B/B/A timings.
  trace169-*.tsv: complete retained activity window, no averaged-away events.
  astra62/TIMELINE.json / CANDIDATE_TIMELINE.json and parent PARENT_BOTH_TIMELINES.json: independent reconstructions.
  trial/DESIGN.json + PERFORMANCE_METRIC.json: fixed-before-timing screen and exactratio definition. FirstT1s and unfavorable results retained.
  contract-review: independent source, allocator, fusion, lifetime and harness reviews.
  ROLES.json: requested model/effort and concrete ownership/interventions. Tool responses did not provide a separate resolved-model field.

Time domains and limits
CUPTI absolute timestamps must be parsed as64-bit integers (Python int is suitable), not roundedJavaScript numbers. Normal caller timing uses its own steady clock; cross-origin timestamps must not be subtracted. Candidate trace contains one apparent896ns same-stream adjacentkernel overlap. Raw values remain; intervalunions prevent double-counting. Its cause was not established. Observerhost overhead changes overlap, especiallyfirstcalls. Only uninstrumentedwhole-token A/B clocks judgeperformance.

PreviouslycompletedFBC/H1/08 are not pendingcommands. CPU08/04/09 and24T rescue remainheld. No newGPUexpertbank is partofthischange. Source/rawmirror duplicates from the incomingProZIP are already in fixedcommit169a2f7eab8d52071661f0da6859d38df4206f8b. EXPORT_SCOPE.json explains which incomingfiles were not duplicated. Root FILES_SHA256 is the publication manifest; incoming ORIGINAL_PACKAGE_FILES_SHA256 is historical.
