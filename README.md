# Meeting analysis model evaluator

This command line harness runs GGUF models through `llama-server`, records JSONL results, mechanically checks meeting analysis outputs, and can ask the Codex CLI for a blind review grounded in the input. It accepts exact request files for ten jobs and keeps the three-message prompt/transcript path for summary and notes.

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
- Sampling defaults to **app parity**: temperature, top-p, top-k, presence penalty and seed use the supplied values; `--min-p` defaults to 0. Extra stages are explicitly neutralized: `repeat_penalty=1`, `dry_multiplier=0`, `xtc_probability=0`, `typical_p=1`, `frequency_penalty=0`, `dynatemp_range=0`, `mirostat=0`, `top_n_sigma=-1`, and `adaptive_target=-1`. This overrides sampler defaults embedded in GGUF metadata.
- The harness requests `verbose` generation settings and checks the server's echoed sampler values, allowing float roundoff. Missing or different values fail the run and prevent a reuse manifest. Each record stores requested `sampler_settings` and the full server-reported `generation_settings`; the manifest retains those reports in trial order. The field names and neutral values were checked against llama.cpp's request schema and settings serializer; a build lacking these fields is refused.
- The request sets `repeat_last_n` to `--n-ctx`, so `llama-server` applies the presence penalty to the prompt as well as the output. An application sampler that penalizes only generated tokens behaves differently.
- Completions stream. By default, the same 12-word span occurring six times within the last 512 whitespace-separated words stops a runaway with `finish="runaway_stopped"`. Detection covers visible output and reasoning separately. Configure `--runaway-repeats` (default 6; 0 disables), `--runaway-ngram` (12), and `--runaway-tail` (512; must fit the repeated spans). The partial output is retained; the server's `/slots` settings are captured before closing the stream. A failed snapshot prevents a reuse manifest. This conservative heuristic can still stop intentional repetition.
- `--parallel N` defaults to 1. It starts `llama-server -np N`, runs at most N trials concurrently, and writes records in case/trial order with unchanged per-trial seeds. Total server context is `N × --n-ctx` to preserve each slot's context budget; KV memory grows accordingly. Parallelism is recorded in the manifest and records. Use serial runs for per-trial speed comparisons.
- In prompt/transcript mode, the third message always ends with `/no_think`. Request-file messages are sent verbatim; the file owns any thinking instructions. For model families with a template thinking switch, also pass it, for example `--chat-template-kwargs '{"enable_thinking": false}'`. `reasoning_chars != 0` is scored as structurally bad unless `score --allow-reasoning` is used.
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

`--judge-model` defaults to the pinned slug `gpt-6.1-sol`, passed to `codex exec -m`.
Judge output records that model and the Codex CLI version. `--judge-concurrency N`
defaults to 1 and bounds concurrent judgments while retaining input order.
HTTP 429 failures retry with 1, 2 and 4 second backoffs (four attempts total);
other failures get the existing single retry. Zero judged trials always exits
nonzero; otherwise errors above `--max-judge-errors` (default 0) exit nonzero.
Even tolerated errors leave affected aggregate findings null.

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

Pass your instruction text with `--prompt-file`. Its exact UTF-8 contents become the first user message. For summary runs, pass an optional GBNF file with `--grammar-file`; notes runs do not accept a grammar. The mechanical scorer reads the prompt to detect copied prompt text and required `## ` sections, so `score --prompt-file` must be the prompt matching the records' intent; `score` stops with an error when a record's `prompt_sha256` differs from that file. The judge reads the applicable prompt to apply its duration and detail contract, using `--summary-prompt-file` or `--notes-prompt-file` according to the intents in the run records. Keep legacy summary and notes scoring invocations separate when their prompt files differ.

## Exact request files

`run --request-file path.json --intent JOB` replaces the prompt/transcript message
builder. The file holds one object or a non-empty list of objects:

```json
{
  "label": "battery-follow-up",
  "maxTokens": 1024,
  "grammar": null,
  "messages": [
    {"role": "system", "content": "Answer only from the meeting. Reply in English."},
    {"role": "user", "content": "Mina said the test passed, but the start date is still open."},
    {"role": "assistant", "content": "The test passed; no start date was chosen."},
    {"role": "user", "content": "Was a start date approved?"}
  ],
  "stream": false
}
```

Case labels must be non-empty and unique within the file. `grammar` contains full
GBNF text or `null`; the example request files embed the same text as their
adjacent grammar.gbnf. Messages accept `system`, `user` and `assistant` roles with
string content. They are sent verbatim, in order, without an added `/no_think`.
`stream` is optional and ignored: the harness always requests a complete response.
Add any model-specific thinking instruction to the file, and use
`--chat-template-kwargs` when required by the model's template.

Each case runs `--trials` times with the same seed sequence. `--label` remains the
model arm and output filename; each record's `case_label` comes from the request.
`--max-tokens` overrides every case's maxTokens. `--grammar-file` overrides every
case's grammar; `--grammar off` disables grammar, including that override.
`--grammar on` requires GBNF from the file or override. Request-file mode does not
accept `--prompt-file`, `--transcript`, `--context`, `--duration`, `--app-user` or
`--language`. The output language belongs in the request messages.

The reuse key covers the request file's exact UTF-8 bytes and every effective
case's messages, grammar, token budget and label. Even changing ignored `stream`
or JSON whitespace changes the key. For these records, `prompt_sha256` hashes the
request file, and `score` and `judge` require the matching `--request-file`; a byte
mismatch is refused before scoring or external judging. Records also store full
`request_messages`, so treat results with the same privacy as inputs. Legacy key
fields retain their meanings; updating evaluate.py changes its own hash and
therefore invalidates earlier keys as it did before.

## Contracts and checks for every job

`run`, `score` and `judge` recognize the same intent registry. New jobs use
request-file input. `score --intent JOB` and `judge --intent JOB` optionally require
all supplied records to have that intent; otherwise the record supplies it.

| Intent | Output | Mechanical checks |
| --- | --- | --- |
| summary | JSON | Existing ordered title/recap/todo envelope, Markdown and prompt checks; optional schema |
| notes | Markdown | Existing title, required H2 sections, duplicate sections and repeated spans; optional string schema |
| speaker_identification | JSON | Schema; assigned names occur in messages after case/accent folding and whitespace collapsing; mapped labels occur in input regardless of case |
| chat | Markdown or text | Common checks and optional string schema; supports earlier chat turns and multiple meetings |
| feedback_analysis | JSON | Schema |
| feedback_evaluation | JSON | Schema; verdict distribution per model arm, intent and case |
| transcript_summary | Plain prose | Non-empty; no Markdown headings or list lines; optional string schema |
| highlight_analysis | XML | Well-formed XML, required root/direct children, unknown elements |
| topic_overview | JSON | Schema |
| transcript_cleanup | JSON or flat text | Schema; turn id coverage, duplicates and fidelity proxies; flat mode string schema and fidelity proxies |

Every output reports `reasoning_chars`, `reasoning_leak`, `prompt_leaks`, `words`,
`empty_visible_text`, `truncated`, `detected_language`, `language_confidence` and `wrong_language`.
`words` counts Unicode word tokens in visible text (JSON string values or XML
text, excluding field names, tags and XML attributes). Zero words sets
`empty_visible_text: true` and makes every intent structurally bad, including
whitespace, zero-width characters and punctuation-only output such as `...`.
Incomplete/non-stop responses, invalid structure and reasoning leakage also make
a run structurally bad.
`finish == "length"` is always a failure. `--allow-reasoning` permits recorded
reasoning and explicit reasoning markers in content. It does not permit malformed
JSON or XML. Reasoning detection uses recorded reasoning and explicit markers
such as `<think>` and channel tokens. Ordinary prose lines beginning `Analysis:`
or `reasoning:` are allowed. These checks do not recognize every kind of drafting prose.

For request files, prompt leak checks compare system-message wording against
non-system messages as evidence. Put instructions in system messages when you
want that distinction; instructions and evidence mixed in one user message
cannot be separated mechanically. As before, `prompt_leaks` is a report for human
review, not by itself a structural failure.

### Schema files

Pass `score --schema-file fixtures/JOB/schema.json`. A schema is required for new
JSON jobs and XML, and for cleanup in either mode. The JSON subset supports
`type` (including a list of types for nullable values), `required`, `properties`,
`items`, `enum`, `minItems`, `maxItems`, `minLength`, `maxLength` and
`additionalProperties: false`. Unknown validation keywords are refused rather
than silently ignored. `title`, `description` and `$schema` are accepted as
annotations; `$schema` does not enable more keywords. Lengths count Python Unicode
characters. `json_valid`, `schema_errors` and `structurally_bad` are reported.
One surrounding JSON code fence is stripped for parsing and reported as
`code_fence: true`; it remains structurally bad.

For Markdown and plain-text kinds, including notes and flat cleanup, a supplied
schema validates the visible text as a string. For example, `type: string`,
`minLength` and `maxLength` apply to that text, and failures appear in
`schema_errors` and make the output structurally bad. A surrounding triple-backtick
code fence, with or without a language label, is removed before measuring visible
text and validating the string; it sets `code_fence: true` and remains structurally bad.

An XML schema lists a `root`, required direct `children` and optional
`allowed_elements`. Without allowed_elements, only the root and required children
are allowed anywhere in the tree. The scorer reports `xml_valid`, `root_matches`,
`missing_elements` and `unknown_elements`. DTD/entity declarations are refused.
This small contract does not validate attributes, ordering or element counts.

`x-eval` holds generic job metadata alongside the JSON schema:

- Speaker mappings: `speakers` names `mappings` (array pointer), `label` and `name`
  (pointers relative to each mapping), plus an optional `input_labels_pattern`
  regular expression. Names are matched after case/accent folding and collapsing
  whitespace runs on both sides, so `Mina  Vale` matches `Mina Vale`. Label
  discovery and comparison are case-insensitive, so `speaker 1` matches
  `Speaker 1`. The default pattern recognizes `Speaker` plus a space and
  digits. `unsupported_names` and
  `unknown_labels` make the output structurally bad. Null names are allowed when
  the schema permits them. Name occurrence is evidence of presence, not proof of
  correct attribution.
- Cleanup: `cleanup` names `input_turns`, `output_turns`, `id` and `text` pointers.
  `output_id` and `output_text` can override the per-turn output pointers. Input
  paths start at `{"messages": [...]}`; output paths start at the parsed output.
  A pointer continuing into a message's content decodes that content as JSON.
  Paths use JSON Pointer escaping (`~1` for `/`, `~0` for `~`) and array indices.
  The supplied structured fixture uses `/messages/1/content/turns`, `/turns`,
  `/id` and `/text`. Input ids must be unique strings or integers. Every source id
  must occur exactly once; `missing_ids`, `extra_ids` and `duplicate_ids` are
  structural failures.
- Flat cleanup: set `mode` to `flat`, `plain` to true, and
  `cleanup.input_text` to a pointer locating the raw text string. See
  `fixtures/transcript_cleanup/request.flat.example.json` and schema.flat.json.
- Verdicts: feedback evaluation finds a property named `verdict` with an enum, or
  the only enum field, searching both `properties` and array `items`. Set
  `verdict` to an output pointer in x-eval when several enum fields are ambiguous.
  Arrays are traversed for every element: `/rows/verdict` selects each row's
  verdict, including through nested arrays. The pointer must locate an enum
  declared by the schema. A misconfigured pointer is reported per scored row as
  `verdict_error` and makes that row structurally bad; scoring continues and
  writes all rows. `verdicts` lists the selected values; `verdict` is retained
  when exactly one value is selected. `verdict_distribution` counts every
  selected value across every array element and trial, including invalid values;
  inspect schema_errors when interpreting the distribution.

Cleanup overlap is multiset token intersection divided by source token count,
with Unicode case/accent folding and punctuation removed. `min_turn_overlap` is
the minimum source-token retention across matched turns; a missing turn scores
zero. `overall_overlap` compares all source and output text; `length_ratio` is
output tokens divided by source tokens. Flat mode reports only the latter two.
These are report-only proxies: deleting a filler lowers overlap, while changing
one negation can preserve a high score. Read the text or use the judge for meaning.
No semantic quality pass is implied by a clean schema.

### Output language

Use `score --expect-language en|de|fr|es|it|pt|nl`. A small stopword-frequency
detector compares the seven languages; it abstains with detected_language null
when fewer than three profile words occur or the leading profiles tie.
`language_confidence` is the winning count's margin over the runner-up, divided
by the winning count, not a calibrated probability. Short names, ids and mixed
languages are often inconclusive. A different detected language sets
`wrong_language`; `--strict-language` makes that a structural failure. Strict mode
requires --expect-language; a detected wrong language also makes output unusable
without strict mode. Abstention still needs human review. When a language is
expected, the scorer checks detectable heading language and copied prompt headings
whose source language differs. Short or ambiguous headings can escape detection. Keep using
`--translated-headings` for the existing summary/notes copied-heading check.

### Evaluating every on-device job

Run a fixture as one arm, then repeat with each comparison model and a different
--label. The JSON fixtures already include a grammar; no --grammar-file is needed.

```sh
python3 evaluate.py run --request-file fixtures/topic_overview/request.example.json \
  --intent topic_overview --server /path/to/llama-server --model /path/to/model.gguf \
  --label candidate --out results/topic --n-ctx 32768 --batch 512 --swa-full off \
  --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5

python3 evaluate.py score --request-file fixtures/topic_overview/request.example.json \
  --schema-file fixtures/topic_overview/schema.json --expect-language en \
  --out results/topic/mechanical.json results/topic/candidate.runs.jsonl

python3 evaluate.py judge --request-file fixtures/topic_overview/request.example.json \
  --schema-file fixtures/topic_overview/schema.json --expect-language en \
  --out results/topic/judge.json results/topic/candidate.runs.jsonl
```

Repeat for the other seven new intent folders, using each folder's schema and
request. feedback_analysis requests German (`--expect-language de`);
transcript_summary requests French (`--expect-language fr`). Test both cleanup
modes. Continue evaluating summary and notes with the existing commands, or
provide their exact requests too. Each new folder includes output.good.txt and
output.bad.txt as hand-written checker examples, not model results. See
[fixtures/long/README.md](fixtures/long/README.md) for 20–30K input-token cases and
output budgets up to 8192 tokens.

For request-file or mixed-intent results, score and judge `summary` is grouped as
intent → case label → model arm, so unrelated cases are not pooled. Single-intent
legacy records keep the existing per-arm summary shape. `grouped_summary` always
provides the grouped shape. The judge uses a job-specific accuracy, compliance,
completeness and language rubric and receives full request messages plus the
optional schema. Legacy summary/notes judgments retain their existing fields;
new jobs add contract_errors, completeness_note, language_note and
language_matches_request. A malformed judgment is an error, and a missing judgment
keeps aggregate findings null.

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

Start with `usable` / `unusable` counts per arm and case. When exactly two arms are
scored, a side-by-side tally prints to stderr; stdout remains JSON. Compare against
the alternative at the same model-file size; a candidate need only improve on that
baseline. Every scored trial has `usable` and `unusable_reasons`. Reasons cover
invalid output, empty visible text, near-empty text (one or two word tokens),
truncation, runaway repetition (a 12-word span occurring at least six times),
reasoning leakage, wrong language or untranslated headings, run errors and
incomplete or mismatched records. `score --expect-no-todos` additionally makes any
summary `user_todos` entry unusable. `--allow-reasoning` retains its existing opt-out.
These mechanical counts are a starting point; legitimate very short answers and
language abstentions still need review. Existing structural and diagnostic fields
remain available.

A structurally clean run passed a narrow envelope and Markdown check. It is not a quality pass.

- `prompt_leaks` lists output spans that share an eight-word run with the prompt but not with the transcript. It covers the summary title and recap body as well as notes, because models copy example sentences into prose that a reader sees, not only into todos. Todos copied whole from the prompt show `text_verbatim_in_prompt`. Also grep the output for your prompt's example names and numbers, which can leak in fragments shorter than eight words.
- Summary rows report `sections`, `missing_sections`, and `title_words`. These are not part of `structurally_bad` for summaries, so an output with only one of the required sections still counts as clean; check them.
- The judge summary leads with typed fields: `structural_leaks` by kind, and todos not assigned to the app user, unsupported, or with a date not stated in the transcript. These are categorical and hold up across judge runs. `unsupported_claims` counts moved from 10 to 13 and from 19 to 11 between runs on identical six-trial inputs, so a difference smaller than that is not a finding. In one comparison the claim counts were 55 and 41, within that noise, while structural leaks were 16 and 0 and every one of nine todos was unsupported; the typed fields decided it.
- The legacy summary/notes judge is not a reliable language check. It has reported zero structural leaks for outputs with English headings inside German text, and it sometimes lists other participants' action items as todos in notes, which have no todo field. A judge parse failure is a missing judgment, not zero findings, and the per-label totals are `null` when any trial failed.
- A run with `reasoning_chars == 0` can still put drafting or self-correction ("I will stick to the transcript spelling...") into the answer. Read the outputs.

## Thinking models

- A GBNF grammar in llama.cpp is not lazy: it applies from the first sampled token, including the reasoning block. With thinking on, summary runs under the JSON grammar came back empty with `finish=stop`. Treat that as a grammar and thinking conflict, not a model verdict. Use `--grammar off` only as a diagnostic.
- Before blaming a model for empty output or truncation, confirm thinking was off: `reasoning_chars` should be 0 in every record. One model was rejected for months for empty output and truncation at 4096 tokens that both came from thinking left on.
- The switch name is family-specific (`enable_thinking` and `thinking` are not interchangeable). `run` uses the chat template embedded in the GGUF through `--jinja`. The template in the model's Hugging Face repository can differ from the embedded one, so a switch that works with one may not work with the other.
- To compare thinking on and off, run grammar-free notes arms that differ only in `--chat-template-kwargs` and score the thinking-on arm with `--allow-reasoning`. Give the thinking-on arm enough `--max-tokens` for reasoning plus the answer, and count a run that spends the whole budget reasoning as a failure.

## Size the context and the benchmark

Size `--n-ctx` from each model's own token count for the full request messages plus the output token budget and a margin. `llama-server`'s `/apply-template` and `/tokenize` endpoints give that count before a run; the rendered template also shows whether the thinking switch took effect (for example, an empty closed think block at the end). After a run, `usage.prompt_tokens` in the records is the same number. Use it for `PROMPT_TOKENS` in `bench.sh speed`.

To inspect a GGUF's architecture, embedded chat template, and control tokens, use `gguf-dump --no-tensors model.gguf` from llama.cpp's `gguf-py` package.

## Speed on a busy host

Record load average, memory pressure, and swap before each arm. If the host is contended, skip `bench.sh` rather than publishing absolute numbers. The `score` summary includes median and minimum `prompt_per_second` and `predicted_per_second` from the quality runs; report those as ratios between arms run in the same session, with the minimums as floors, since both arms carried the same load. This does not hold for an arm reused from another session, and KV allocation still needs a quiet host.

Trials from `--parallel N` with N > 1 retain raw timings for diagnostics but are
excluded from these speed summaries because slots contend for resources. Early
stops may have no final usage or timings. Judge concurrency, streamed runaway
cancellation, avoiding a second read of results to build the manifest, and promptly
terminating the owned server on interruption also reduce avoidable waiting or I/O.

## Reuse a stored arm

Re-running the same incumbent for every candidate is slow. After all trials finish without an error, `run` writes `<label>.manifest.json` with a `reuse_key` and its SHA-256; a new run first removes any earlier manifest for its label, so a failed or interrupted run leaves none. The key covers the host, the `llama-server` version and the hashes of the binary and the shared libraries beside it (after resolving a symlinked `--server`), the `evaluate.py` hash, the model hash, the prompt, grammar, and transcript hashes, a hash of the exact messages sent (so it follows the context file, or `--duration`, `--app-user`, and `--language` when there is none), the template kwargs, sampling, `--n-ctx`, `--max-tokens`, `--batch`, `--gpu-layers`, `--require-metal`, `--swa-full`, the starting seed, and the trial count. Hashing the model reads the whole file once.

The key now also includes every requested sampler control and the runaway-stop
configuration, plus `parallel` when N > 1; serial mode adds no parallel key field.
This changes old keys, as does the harness file hash. Runtime settings are kept
outside the key, since `--key-only` starts no server; only settings verified against
the keyed sampler controls can produce a reuse manifest. Disabled stages' secondary
parameters and other server-reported settings remain in the full runtime report.

To check whether a stored arm can stand in, run the same `run` command with `--key-only`, which prints the requested reuse key without starting the server, and compare `reuse_key_sha256` with the stored manifest's. Reuse the stored `*.runs.jsonl` only when they are equal. Any difference, including an edited prompt or a newer `evaluate.py`, means running it again. Always re-run `score` and `judge`, and never use a reused arm's timings as speed evidence.

## Shell notes

`bench.sh` runs under the stock macOS `/bin/bash` 3.2. For longer sequences, put the commands in a bash script file rather than pasting them into an interactive or agent shell; zsh differs from bash in array handling, and `$(jobs -p)` came back empty in a non-interactive zsh. Stock macOS has no `timeout`; use `gtimeout` from coreutils or `curl --max-time`.

## Get a transcript

The request fixtures contain short invented meetings. For a longer transcript, any public meeting recording works. For a video with automatic captions, download subtitles without downloading the media, then convert them:

```sh
yt-dlp --write-auto-sub --sub-langs en --sub-format vtt --skip-download VIDEO_URL
python3 evaluate.py captions VIDEO_FILE.en.vtt --out transcript.txt
```

`captions` drops the WebVTT header, cue timings, and inline tags, decodes entities, and removes the consecutive repeats that rolling automatic captions produce. For a non-English meeting, request the video's original caption language (for example `--sub-langs de`), not a machine translation.

## Privacy

`judge` sends the prompt file contents (`production_prompt`), the `--context` file contents when a run stored `app_supplied_context`, the transcript, and the model output to the Codex CLI, an external provider. Use it only with content you are permitted to share.

With `--request-file`, judge instead sends the full request messages, model output, optional schema and expected language to the external provider. The fixtures use invented names and meetings; replace them only with content you may share.

The judge is nondeterministic, so counts vary from run to run and are evidence for a human, not a score.
