# Hardware baseline archive

The user paused optimization on 2026-11-05. This repository preserves an E5-2630L v2 dual-socket, 512 GiB, PG500-216 150 W/ECC baseline. Archival checks are allowed. Do not run benchmarks, build optimization candidates, restart an API, or deploy experiments unless the user explicitly resumes that work.

Read README.md and profiles/e5-2630lv2-512gb/CHECKPOINT.json first. accepted/ contains the source files used by the adopted runtime. lab/ also contains historical and unadopted experiments; presence here does not authorize executing them or make them production settings. Do not follow old experiment queues in archived RESUME files.

Preserve raw evidence and frozen model/calibration hashes. New hardware measurements belong in a new dated profile/study. Do not refit an old model and reuse its heldout evaluation.

Do not commit model weights, API keys, authentication files, real conversation/server logs, CUDA toolkit libraries, or generated build objects. Use an allowlist and scripts/verify_archive.py before publishing. The runtime release is a separate copy of already-built project binaries, with its own checksums; it is not a new optimization build.

Long model startups and downloads are user-run. Monitoring is handled by the existing local service; do not replace it with repeated assistant polling. GPU defaults remain 150 W, ECC enabled, Standard fans, maximum 70%.
