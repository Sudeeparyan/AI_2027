# Application verification — 28 September 2026

10 unittest tests passed. Scope: temporary quota retry and fallback; permanent authentication failure; all providers unavailable; actual HTTP request/response against a local mock server; insecure remote URL rejection; resume after model change; deliberate refresh; invalid JSON preserving canonical content; apply backups and stale-source rejection; run locking; minimum speaker-note validation. Some tests check multiple conditions.

Source ingestion indexed all 11 attachments. Dry run prepared all five week 1 content jobs without API calls. One representative week was rebuilt after adding per-week reports.

No live cloud provider calls were made and no credentials were supplied. Provider endpoint compatibility must be smoke-tested with the selected model before a paid full-course run. Tests establish the implemented retry/checkpoint behavior, not guaranteed factual quality or distributed scalability.
