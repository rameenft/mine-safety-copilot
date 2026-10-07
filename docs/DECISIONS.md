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

## M8: manual review turned into scoring rules
Reading the 24 cached answers by hand (not just the score) found issues the rule-based scorer
couldn't see: agg-05/06/07 didn't flag the 3,000-row sample up front, agg-06 called it a
"representative random sample" (it isn't, since every fatality is kept), and hyb-01/02 said
"involving conveyors/berms" for counts that are keyword matches. Reading the 12 matched fatal
narratives confirmed that some only *mention* the keyword (e.g. a crane carrying a conveyor belt
hit a power line). Fixes:
- Golden items carry phrase rules: `lead_phrases` (must be in the opening paragraph),
  `required_phrases`, and `forbidden_phrases` (a "not" within 30 chars before it is allowed, so
  "should not be considered representative" passes).
- The system prompt now requires the sample caveat in the first sentence and "mentioning" for keyword counts.
- **Partial answers** (new type, 6 questions): when only part of a question is in scope, answer
  that part and end with "Not covered:" plus a pointer from a fixed list in the prompt (Part 46
  training, Part 100 penalties, Part 57, coal Parts 70-75, OSHA). The pointers live in the system
  prompt, so the grounding check accepts their numbers. The model can't make up other references.
Old answers score 19/24 under the new rules and the new prompt scores 29/30. hyb-01 still says
"involving" and is left as a visible failure rather than prompt-tuned to pass a single question.
M5 answers are archived in `evals/runs/gemini-3.8-flash-m5/` for comparison.

## Dated rule versions: handled in the tool, not the prompt
Part 56 holds two dated pairs: 56.5001/56.5005 ("required until April 7, 2026") and
56.5001T/56.5005T ("As of April 8, 2026"). Two new golden items (reg-10, part-07) forbid citing
the expired version. Three attempts, each run live on just those two questions:
1. A prompt rule ("cite the version in force today"): the model cited the expired versions. It
   doesn't know today's date, so "until April 7, 2026" looked current.
2. A `status` label ("EXPIRED … use 56.5005T") on tool results: still cited the expired version,
   after ~19k output tokens of reasoning. The model overrode an explicit label it disagreed with.
3. Hide expired sections from search, have `get_regulation` on an expired ID return its in-force
   replacement with a note, and put today's date in the system prompt: reg-10 passes.
Lesson: facts the model must not get wrong go in code, not instructions. Trade-off: the expired
text can no longer be quoted, even for questions about 2021-2024 accidents that happened under it.
part-07 ("silica limit at my quarry?") now refuses outright and points to Part 60 instead of
giving a partial answer citing 56.5001T. That's safe but doesn't match the expected behaviour, so
it stays a visible failure. The other 30 questions were not re-run after these prompt and tool
changes. `evals/reports/agent.md` is the 30-question run from before them.

## MCP server reuses the agent's tools, not a parallel implementation
`mcp_server.py` uses the low-level `mcp` 2.x `Server` and serves `TOOL_SPECS` as the tool schemas,
running each call through `call_tool`. FastMCP/MCPServer was skipped because it derives schemas
from function signatures, which would duplicate (and drift from) the schemas the agent uses. All
tools are annotated read-only. Guardrails that live in the tool layer carry over: the allow-listed
stats filters, the read-only DB, and expired sections hidden and redirected. The grounding check and
refusal rules live in the agent loop, so over MCP they're only advisory, sent as server
`instructions` that reuse the agent's system prompt. This is the case for putting safety-critical behaviour
in tools rather than prompts. Search now over-fetches by the number of hidden expired sections so
it still returns k results. `mcp` is an optional extra, and its tests skip when it isn't installed.
