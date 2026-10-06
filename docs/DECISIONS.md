# Decisions

## Gemini free tier behind a provider interface
Free tier keeps the build at $0. All model calls go through `llm.py`, so moving to Claude
(Anthropic API or Bedrock) is a config change, not a rewrite.

## Lean, incremental dependencies
`pyproject.toml` only lists what the current milestone needs. Keeps installs fast and makes
each dependency's purpose obvious in the git history.

## Model: gemini-3.8-flash
Newest stable (non-preview) Flash model on the free tier as of 2026-10-06; verified with a test call.
Pinned in `.env` via `GEMINI_MODEL`; fallback is `gemini-2.5-flash` if free-tier quota runs out.

## Surface scope = 3 subunits
Kept STRIP/QUARRY/OPEN PIT, MILL/PREP PLANT, DREDGE. Dropped "SURFACE AT UNDERGROUND" because Part 57,
not Part 56, governs it — mixing them would let the bot cite the wrong part.

## Derived SEVERITY column
`DEGREE_INJURY` has 11 cryptic codes ("DYS AWY FRM WRK & RESTRCTD ACT"). A 6-value `SEVERITY`
column gives the tools a small, validatable filter vocabulary.

## Regulations in SQLite too
Sections are stored in a `regulations` table so `get_regulation(section_id)` is an exact lookup,
independent of the vector index. Suffixed IDs (56.5001T) are kept as distinct sections.

## Fixtures include must-reject rows
The accident fixture mixes 100 valid rows with coal/2019/underground/blank-narrative rows, so the
loader test proves each filter works instead of only checking the happy path.

## Retrieval: dense default, BM25 fallback, hybrid measured
Golden-set eval (13 reg/hybrid Qs): BM25 R@5 0.85, dense (gemini-embedding-001, 768-d) 0.92,
equal-weight RRF hybrid only R@1 0.62, because BM25 noise dragged good dense hits down. Dense-weighted (2×)
RRF reaches 0.85 but still trails dense, so dense is the default. BM25 runs with no API key
(tests/CI) and section IDs named in a query ("56.14107") are pinned via exact match. n=13 is small:
re-check once the M4 agent sends decomposed sub-queries instead of mixed hybrid questions.

## Embeddings stored as a .npy matrix, not a vector DB
459 chunks × 768 floats ≈ 1.4 MB; brute-force dot product is instant. A vector DB would add a
dependency with no benefit at this scale. Free tier caps ~100 texts/min, so the build retries on 429.

## Agent: allow-listed stats tool, not text-to-SQL
`accident_stats` takes typed filters (severity enum, year range, substring filters, narrative
keywords ORed) plus an allow-listed `group_by`/`metric`; all values are bound parameters and the DB
is opened read-only. Text-to-SQL is more flexible but its failures (bad joins, injection,
silently wrong filters) are hard to detect; every golden agg/hybrid number is reachable this way.
Errors come back as `{"error": ...}` so the model can self-correct (e.g. year 2018 → refuse).

## Grounding check after the answer
`ground()` compares every cited § section, 12-digit document number and loose number in the
answer with tool output (plus the question and the scope years). Unsupported ones are dropped from
`citations` and listed in `ungrounded`. Number words in regulations ("seven feet") count as
evidence for digits. It can't verify paraphrased claims; M5's eval checks content.

## Provider keeps the raw model turn
Gemini 3 requires thought signatures on function-call turns to be sent back, so `Reply.raw` stores
the native turn and is replayed verbatim; the neutral message format stays SDK-free.

## Free tier: 20 requests/day on gemini-3.8-flash
Discovered during the M4 smoke test (4 Qs ≈ 15 calls). A daily-quota 429 fails fast instead of
retrying. The M5 eval (24 Qs × ~3 calls) needs either `GEMINI_MODEL=gemini-2.5-flash` (higher free
quota) or a run split across days.

## M5 eval: deterministic scoring, no LLM judge
`evals/eval_agent.py` scores answers with rules, not a second model. A fact counts when ≥60% of
its content words (crudely stemmed) appear in the answer, so "must be guarded" matches "shall be
guarded". A value has to appear as a number or whole word, and a section only counts if it
survives the grounding check. Rules are cheap, repeatable and need no extra quota. The trade-off
is that a paraphrase sharing few words gets marked wrong; the failures list in the report makes
those easy to audit by hand.

## Eval runs are cached per model
`--model` picks the model per run; answers are cached in `evals/runs/<model>/` (gitignored), so a
run cut off by quota resumes, and `--rescore` rebuilds the report offline. Calls are throttled
(`--gap`, default 13s for the 5 requests/minute free tier).

## M5 eval switched to gemini-3.8-flash (paid key)
2.5-flash's free tier turned out to be 20 requests/day as well, so billing was enabled and the full
run uses gemini-3.8-flash, the model M4 was built on (~108k input / 13k output tokens all-in).
The first run scored 0.96 and surfaced two agent bugs, both fixed:
- reg-05: the model used all 5 tool turns, then returned empty text on the tools-withheld turn.
  The final turn now adds a user nudge ("Tool budget used up. Answer now…").
- agg-05/06: the grounding check flagged markdown list markers ("7. TX: 263") as invented numbers.
  `ground()` strips list markers first; the eval re-grounds cached traces so checker fixes need no
  new API calls.
After the fixes: 24/24 on every metric. The set is small and in-house, so it's a regression gate.

## Demo UI is Streamlit, logic lives outside it
Streamlit is an optional `[demo]` extra so the core install stays lean. `app.py` only lays out the
page; token/latency metering, error messages and the eval-report summary live in `demo.py` and are
unit-tested without Streamlit. Tests inject `provider`/`tools` via `st.session_state`, so
`streamlit.testing` drives the real render path with `FakeProvider` and the fixture DB, offline.
