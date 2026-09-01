# Meeting recap evaluation harness

This command line harness runs GGUF models through `llama-server` with a three-message meeting transcript request, records JSONL results, mechanically checks recap or notes structure, and can ask the Codex CLI for a blind transcript-grounded review.

## Requirements

- Python 3.11 or newer, using only the standard library
- A llama.cpp build containing `llama-server` and `llama-bench`
- A signed-in Codex CLI for `judge`
- macOS with Metal. This is the only tested backend, and `run` refuses to continue unless an Apple Metal device is positively identified.

## Quick start

Run the built-in checks:

```sh
python3 evaluate.py self-test
```

Run two models against one transcript. Replace the binary, model, and transcript paths with local files:

```sh
python3 evaluate.py run \
  --prompt-file fixtures/summary/prompt.txt \
  --grammar-file fixtures/summary/grammar.gbnf \
  --intent summary --server /path/to/llama/bin/llama-server \
  --model /path/to/candidate.gguf --label candidate \
  --transcript /path/to/transcript.txt --context fixtures/context.example.txt \
  --out results --duration "42 minutes" --app-user "Casey Rivera" \
  --language English --n-ctx 16384 --max-tokens 4096 --batch 512 \
  --swa-full off --temperature 0.7 --top-p 0.8 --top-k 20 \
  --presence-penalty 1.5

python3 evaluate.py run \
  --prompt-file fixtures/summary/prompt.txt \
  --grammar-file fixtures/summary/grammar.gbnf \
  --intent summary --server /path/to/llama/bin/llama-server \
  --model /path/to/incumbent.gguf --label incumbent \
  --transcript /path/to/transcript.txt --context fixtures/context.example.txt \
  --out results --duration "42 minutes" --app-user "Casey Rivera" \
  --language English --n-ctx 16384 --max-tokens 4096 --batch 512 \
  --swa-full off --temperature 0.7 --top-p 0.8 --top-k 20 \
  --presence-penalty 1.5
```

`--trials` defaults to 3; `run` refuses to overwrite an existing `<label>.runs.jsonl`; `--context` replaces the `--duration`/`--app-user`/`--language` line in the third message, though those flags remain required and should match the file; and model families with a template thinking switch need `--chat-template-kwargs '{"enable_thinking": false}'` or their equivalent because `reasoning_chars != 0` is scored as structurally bad.

Score and judge those records:

```sh
python3 evaluate.py score \
  --prompt-file fixtures/summary/prompt.txt \
  --transcript /path/to/transcript.txt --out results/mechanical.json \
  results/candidate.runs.jsonl results/incumbent.runs.jsonl
```

`judge` sends the prompt file contents (`production_prompt`), the `--context` file contents when a run stored `app_supplied_context`, the transcript, and the model output to the Codex CLI, an external provider. Use it only with content you are permitted to share.

```sh
python3 evaluate.py judge \
  --summary-prompt-file fixtures/summary/prompt.txt \
  --transcript /path/to/transcript.txt --out results/judge.json \
  results/candidate.runs.jsonl results/incumbent.runs.jsonl
```

Measure speed or KV allocation after choosing explicit settings:

```sh
PROMPT_TOKENS=8000 REPEATS=3 BATCH=512 \
  bash bench.sh speed /path/to/llama/bin results/bench \
  candidate=/path/to/candidate.gguf incumbent=/path/to/incumbent.gguf

CONTEXTS="16384" BATCH=512 SWA_FULL=off \
  bash bench.sh kv /path/to/llama/bin results/bench \
  candidate=/path/to/candidate.gguf incumbent=/path/to/incumbent.gguf
```

`python3 evaluate.py parse-kv results/bench/kv_*.log` parses the KV allocation from the kv logs.

## Summary envelope contract

Summary output is one JSON object with exactly these keys in this order: `title`, `recap_markdown`, and `user_todos`. `title` and `recap_markdown` are strings. `user_todos` is a list of objects with exactly `text`, a string, and `dueDate`, either a string or `null`. The included summary grammar enforces this shape.

## Use your own prompt and grammar

Pass your instruction text with `--prompt-file`. Its exact UTF-8 contents become the first user message. For summary runs, pass an optional GBNF file with `--grammar-file`; notes runs do not accept a grammar. The mechanical scorer reads the prompt to detect copied todo text or required notes headings. `score --prompt-file` must be the prompt matching the records' intent because it reads `## ` headings as required notes sections. The judge reads the applicable prompt to apply its duration and detail contract, using `--summary-prompt-file` or `--notes-prompt-file` according to the intents in the run records.

## Get a transcript

No transcript is included. Any public meeting recording works. For a video with automatic captions, download subtitles without downloading the media:

```sh
yt-dlp --write-auto-sub --sub-lang en --skip-download VIDEO_URL
```

Convert the subtitle file to plain text before running the harness.

## Privacy

`judge` sends the prompt file contents (`production_prompt`), the `--context` file contents when a run stored `app_supplied_context`, the transcript, and the model output to the Codex CLI, an external provider. Use it only with content you are permitted to share.

The judge is nondeterministic, so counts vary from run to run and are evidence for a human, not a score.
