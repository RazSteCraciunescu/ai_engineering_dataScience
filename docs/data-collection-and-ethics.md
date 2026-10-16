# Data collection and ethics

The collectors read local exports to make the project deterministic and public-safe. The same principles apply to real collection work: identify the source and timestamp, preserve original identifiers, minimize fields, respect terms and robots rules, apply rate limits, and obtain privacy/legal review before collecting personal or restricted data.

A production collector also needs retries, backoff, checksums, scheduling, change detection and dead-letter handling. The HTML collector here intentionally remains small and transparent.
