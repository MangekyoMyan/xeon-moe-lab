xeon-moe-lab abs operator review / next mixed experiment
Pinned commit: 31b1292b2767618a28ef6e3f1c9c4450b1fc21be

Start with REVIEW_AND_NEXT_STEP_JA.txt.
For the Astra on the actual Xeon machine, read ASTRA_NEXT_ACTION_JA.txt.
NEXT_RUN_SPEC.json fixes the proposed experiment before any new real timings.
next_experiment/ contains candidate sources and the generated case.
evidence/, contracts/, priority/ retain independent audits and source manifests.
checks/ and VERIFICATION.json separate actual local checks from unperformed engine runs.

The mixed harness has been syntax-checked; its ggml link and Xeon performance run remain unperformed.
Use the existing outer guard on the real machine. No CPU library or production profile change is needed for this isolated test.
No generated ELF or model weights are included. Synthetic test times are not performance results.
