# Meeting analysis model evaluator

This command line harness runs GGUF models through `llama-server` with a three-message meeting transcript request, records JSONL results, mechanically checks recap or notes structure, and can ask the Codex CLI for a blind transcript-grounded review.

## Requirements

- Python 3.11 or newer, using only the standard library
- A llama.cpp build containing `llama-server` and `llama-bench`
- A signed-in Codex CLI for `judge`
- macOS with Metal. This is the only tested backend, and `run` refuses to continue unless `llama-server --list-devices` lists an Apple Metal device.

Sandboxed shells, including some coding-agent sandboxes, can hide Metal from the process. llama.cpp then falls back to CPU and prints plausible but wrong throughput. When the gate fails inside a sandbox, run outside it; do not pass `--no-require-metal` to get past it.

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
  --swa-full off --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5

python3 evaluate.py run \
  --prompt-file fixtures/summary/prompt.txt \
  --grammar-file fixtures/summary/grammar.gbnf \
  --intent summary --server /path/to/llama/bin/llama-server \
  --model /path/to/incumbent.gguf --label incumbent \
  --transcript /path/to/transcript.txt --context fixtures/context.example.txt \
  --out results --duration "42 minutes" --app-user "Casey Rivera" \
  --language English --n-ctx 16384 --max-tokens 4096 --batch 512 \
  --swa-full off --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5
```

Notes on `run`:

- `--trials` defaults to 3, and three is the minimum worth reading. One model emitted 1, 3, and 10 todos on three seeds of the same input; a single run would have hidden that.
- `run` refuses to overwrite an existing `<label>.runs.jsonl`.
- `--min-p` defaults to 0 and is always sent. Without it, `llama-server` applies its own default of 0.05, which most publisher settings do not use.
- The request sets `repeat_last_n` to `--n-ctx`, so `llama-server` applies the presence penalty to the prompt as well as the output. An application sampler that penalizes only generated tokens behaves differently.
- The third message always ends with `/no_think`. For model families with a template thinking switch, also pass it, for example `--chat-template-kwargs '{"enable_thinking": false}'`. `reasoning_chars != 0` is scored as structurally bad unless `score --allow-reasoning` is used.
- When every trial finishes without an error, `run` also writes `<label>.manifest.json`; see [Reuse a stored arm](#reuse-a-stored-arm).

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

Pass your instruction text with `--prompt-file`. Its exact UTF-8 contents become the first user message. For summary runs, pass an optional GBNF file with `--grammar-file`; notes runs do not accept a grammar. The mechanical scorer reads the prompt to detect copied prompt text and required `## ` sections, so `score --prompt-file` must be the prompt matching the records' intent. The judge reads the applicable prompt to apply its duration and detail contract, using `--summary-prompt-file` or `--notes-prompt-file` according to the intents in the run records.

## Context and response language

When `--context` is given, the file's contents replace the generated line built from `--duration`, `--app-user`, and `--language`. Those flags are still required but are not sent. The response language the model sees is whatever the context file says, so a non-English arm needs its own context file; changing `--language` alone does nothing.

Test at least one non-English transcript. Wrong-language and mixed-language output is a defect class an English-only evaluation cannot see. `fixtures/context.de.example.txt` is a German context that asks for German throughout, headings included. Score such an arm with `--translated-headings`:

```sh
python3 evaluate.py score --translated-headings \
  --prompt-file fixtures/notes/prompt.txt \
  --transcript /path/to/german.txt --out results/de/notes/mechanical.json \
  results/de/notes/candidate.runs.jsonl results/de/notes/incumbent.runs.jsonl
```

With that flag, an H2 heading copied verbatim from the prompt is listed in `untranslated_headings` and makes the run structurally bad, and notes need as many sections as the prompt names instead of its literal English names, which correctly translated output can never match. The check only catches copied prompt headings. Read the output for English prose, English sub-headings, or a heading that happens to be the same word in both languages. If your prompt names its sections in English while the context asks for another language, the prompt itself is ambiguous, and both arms usually show it; compare the arms before blaming one model.

## Read the results

A structurally clean run passed a narrow envelope and Markdown check. It is not a quality pass.

- `prompt_leaks` lists output spans that share an eight-word run with the prompt but not with the transcript. It covers the summary title and recap body as well as notes, because models copy example sentences into prose that a reader sees, not only into todos. Todos copied whole from the prompt show `text_verbatim_in_prompt`. Also grep the output for your prompt's example names and numbers, which can leak in fragments shorter than eight words.
- Summary rows report `sections`, `missing_sections`, and `title_words`. These are not part of `structurally_bad` for summaries, so an output with only one of the required sections still counts as clean; check them.
- The judge summary leads with typed fields: `structural_leaks` by kind, and todos not assigned to the app user, unsupported, or with a date not stated in the transcript. These are categorical and hold up across judge runs. `unsupported_claims` counts moved from 10 to 13 and from 19 to 11 between runs on identical six-trial inputs, so a difference smaller than that is not a finding. In one comparison the claim counts were 55 and 41, within that noise, while structural leaks were 16 and 0 and every one of nine todos was unsupported; the typed fields decided it.
- The judge is not a language check. It has reported zero structural leaks for outputs with English headings inside German text, and it sometimes lists other participants' action items as todos in notes, which have no todo field. A judge parse failure is a missing judgment, not zero findings, and the per-label totals are `null` when any trial failed.
- A run with `reasoning_chars == 0` can still put drafting or self-correction ("I will stick to the transcript spelling...") into the answer. Read the outputs.

## Thinking models

- A GBNF grammar in llama.cpp is not lazy: it applies from the first sampled token, including the reasoning block. With thinking on, summary runs under the JSON grammar came back empty with `finish=stop`. Treat that as a grammar and thinking conflict, not a model verdict. Use `--grammar off` only as a diagnostic.
- Before blaming a model for empty output or truncation, confirm thinking was off: `reasoning_chars` should be 0 in every record. One model was rejected for months for empty output and truncation at 4096 tokens that both came from thinking left on.
- The switch name is family-specific (`enable_thinking` and `thinking` are not interchangeable). `run` uses the chat template embedded in the GGUF through `--jinja`. The template in the model's Hugging Face repository can differ from the embedded one, so a switch that works with one may not work with the other.
- To compare thinking on and off, run grammar-free notes arms that differ only in `--chat-template-kwargs` and score the thinking-on arm with `--allow-reasoning`. Give the thinking-on arm enough `--max-tokens` for reasoning plus the answer, and count a run that spends the whole budget reasoning as a failure.

## Size the context and the benchmark

Size `--n-ctx` from each model's own token count for the full three-message request plus `--max-tokens` and a margin. `llama-server`'s `/apply-template` and `/tokenize` endpoints give that count before a run; the rendered template also shows whether the thinking switch took effect (for example, an empty closed think block at the end). After a run, `usage.prompt_tokens` in the records is the same number. Use it for `PROMPT_TOKENS` in `bench.sh speed`.

To inspect a GGUF's architecture, embedded chat template, and control tokens, use `gguf-dump --no-tensors model.gguf` from llama.cpp's `gguf-py` package.

## Speed on a busy host

Record load average, memory pressure, and swap before each arm. If the host is contended, skip `bench.sh` rather than publishing absolute numbers. The `score` summary includes median and minimum `prompt_per_second` and `predicted_per_second` from the quality runs; report those as ratios between arms run in the same session, with the minimums as floors, since both arms carried the same load. This does not hold for an arm reused from another session, and KV allocation still needs a quiet host.

## Reuse a stored arm

Re-running the same incumbent for every candidate is slow. After all trials finish without an error, `run` writes `<label>.manifest.json` with a `reuse_key` and its SHA-256; a failed or interrupted run leaves no manifest. The key covers the host, the `llama-server` version and the hashes of the binary and the shared libraries beside it (after resolving a symlinked `--server`), the `evaluate.py` hash, the model hash, the prompt, grammar, and transcript hashes, a hash of the exact messages sent (so it follows the context file, or `--duration`, `--app-user`, and `--language` when there is none), the template kwargs, sampling, `--n-ctx`, `--max-tokens`, `--batch`, `--gpu-layers`, `--require-metal`, `--swa-full`, the starting seed, and the trial count. Hashing the model reads the whole file once.

To check whether a stored arm can stand in, run the same `run` command with `--key-only`, which prints the manifest without starting the server, and compare `reuse_key_sha256` with the stored manifest's. Reuse the stored `*.runs.jsonl` only when they are equal. Any difference, including an edited prompt or a newer `evaluate.py`, means running it again. Always re-run `score` and `judge`, and never use a reused arm's timings as speed evidence.

## Shell notes

`bench.sh` runs under the stock macOS `/bin/bash` 3.2. For longer sequences, put the commands in a bash script file rather than pasting them into an interactive or agent shell; zsh differs from bash in array handling, and `$(jobs -p)` came back empty in a non-interactive zsh. Stock macOS has no `timeout`; use `gtimeout` from coreutils or `curl --max-time`.

## Get a transcript

No transcript is included. Any public meeting recording works. For a video with automatic captions, download subtitles without downloading the media, then convert them:

```sh
yt-dlp --write-auto-sub --sub-langs en --sub-format vtt --skip-download VIDEO_URL
python3 evaluate.py captions VIDEO_FILE.en.vtt --out transcript.txt
```

`captions` drops the WebVTT header, cue timings, and inline tags, decodes entities, and removes the consecutive repeats that rolling automatic captions produce. For a non-English meeting, request the video's original caption language (for example `--sub-langs de`), not a machine translation.

## Privacy

`judge` sends the prompt file contents (`production_prompt`), the `--context` file contents when a run stored `app_supplied_context`, the transcript, and the model output to the Codex CLI, an external provider. Use it only with content you are permitted to share.

The judge is nondeterministic, so counts vary from run to run and are evidence for a human, not a score.
