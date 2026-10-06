# CLAUDE.md — Mine Safety Copilot

Grounded safety Q&A for US **surface metal/nonmetal** mines, answering only from
30 CFR Part 56 and MSHA accident records (2021–2024). Portfolio project for an
Avathon "Associate AI Engineer, Physical AI" interview. One-day build, Claude Pro (limited usage).

## Working rules
- One milestone at a time: plan (5–10 lines) → wait for OK → build → stop.
- Short explanations (5–8 lines); deeper only on "explain more".
- After each milestone: 3–5 line walkthrough, 3-point review checklist, 2 quiz questions, 1-line talking point.
- Save tokens: never print large files/data (`head`, `wc`, small samples); don't re-read files just written.
- Ask before: creating/pushing the GitHub repo, or anything that spends API money (give estimate).
- Never commit secrets (`.env`) or raw/processed data. Fixtures live in `tests/fixtures/`.
- Update `## Status
**Done:**
- M0: venv, skeleton, config, git, public repo https://github.com/rameenft/mine-safety-copilot. Model `gemini-3.8-flash` (key verified).
- M1: `ingest/` (accidents.py, regulations.py, build.py, fixtures.py) → `data/processed/mine_copilot.db` with tables `accidents` (3,000), `mines` (1,410), `regulations` (422 sections). Fixtures + 10 passing tests. `docs/DATA_CARD.md`.

- M2: `evals/golden_set.yaml` — 24 Qs (9 reg / 7 agg / 4 hybrid / 4 refuse), generated + verified by `evals/build_golden.py` (facts verbatim in section text; numbers from SQL). `tests/test_golden_set.py` (13 tests passing).

- M3: `retrieval/` (index.py, embed.py, build.py) — 459 chunks, BM25 + Gemini dense (768-d .npy) + weighted RRF; default mode dense. `evals/eval_retrieval.py` → `evals/reports/retrieval.md` (dense R@5 0.92, MRR 0.92; BM25 0.85). 20 tests passing (no network).

- M4: `llm.py` (Provider protocol, GeminiProvider, FakeProvider) + `agent/` (tools.py: search_regulations, get_regulation, allow-listed accident_stats; loop.py: ≤5 tool turns, refusal prefix, grounding check; CLI `python -m mine_copilot.agent "q" [--json]`). 31 tests passing offline. Live smoke: 4/4 golden Qs correct.

- M5: `evals/eval_agent.py` (rule-based scoring, per-model cache in `evals/runs/`, `--rescore` offline, token tracking) → `evals/reports/agent.md`. gemini-3.8-flash on a paid key: 24/24 answer acc, citation recall, refusal acc, grounded (first run 0.96; fixed empty final answer + list-marker grounding false positive). 36 tests passing.

**Next:** M6.

**Quota:** free tier is 20 req/day per model (3.8-flash and 2.5-flash); billing now enabled ($5 credit). M5 full run ≈ 108k in / 13k out tokens.

**Decisions:** see `docs/DECISIONS.md`. Rebuild data: `python -m mine_copilot.ingest.build`; index: `python -m mine_copilot.retrieval.build` (~5 min, free-tier rate limit; `--no-embed` for BM25 only). `chunk_section()` (1,500 chars, paragraph split) is ready for M3 indexing. `SEVERITY` column is the severity filter (fatal / permanent_disability / lost_time / restricted_duty / no_lost_time / other).
