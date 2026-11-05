Current user operating policy (2026-11-01):

- User paused optimization on 2026-11-05. Preserve the current E5-2630L v2 configuration as a hardware-specific baseline. Do not start further optimization builds, benchmarks, or deploy experimental scheduling until the user asks to resume. Archival, GitHub setup/support, and existing operational monitoring remain in scope. CPU replacement should begin with fresh measurements; reuse validated kernels and measurement tools rather than discarding prior work.

- Give the user commands for long downloads, model startups/preloads and other lengthy operations. Do not launch them on the user's behalf unless explicitly requested again.
- Do not spend model turns polling normal temperatures, download progress or model loading. Local software should monitor and notify only on meaningful events.
- Thermal monitoring implementation: thermal-monitor-20261101. Persistent sensor state: /var/lib/xeon-watch/latest.json. Notification receipts: /var/lib/xeon-watch-notify. Check service status rather than assuming a monitor is running.
- A [XEON-WATCH ALERT] is an event report. Read the latest snapshot once if needed, report concisely, and do not start benchmarks, restart models, or repeat idle polling. A SELFTEST requires only brief receipt confirmation.
- Preserve GPU150W default, ECC enabled, Standard fans and max70% unless the user explicitly changes these policies. Do not switch fans to Full.
- Real model comparisons in ram-model-20261031 were cancelled by the user. Do not resume that controller. Existing production API may be active; inspect it before other GPU work. Do not start an inactive API automatically under the latest user policy.
- Preserve original model files, frozen engines, evidence and private credentials. Keep user-requested work separate from old, completed experiment instructions.

- Subagent preference (2026-11-03): User permits gpt-6-astra with xhigh reasoning for demanding analysis, and prefers gpt-6.1-terra for reviews and log summaries. This is permission/preference, not a requirement to spawn agents for every task. Keep important implementation edits with the primary agent.
