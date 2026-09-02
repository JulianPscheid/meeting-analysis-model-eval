This PR introduces a compare subcommand to valuate.py to streamline model evaluation workflows.

Previously, comparing multiple model runs required manual inspection of the raw JSON artifacts from the score and judge stages. This subcommand parses those artifacts and emits a structured, side-by-side Markdown report. It aggregates mechanical validation rates, generation throughput, and Codex judge verdicts into a unified view.

I've kept the implementation dependency-free, strictly relying on the standard library to maintain the existing constraints. It gracefully degrades if judge data is omitted, and defaults to stdout to support standard POSIX pipe chains.

Self-tests have been updated to cover the new Markdown rendering paths.
