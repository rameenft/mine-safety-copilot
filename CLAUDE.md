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

**Next:** M3 — retrieval index over `chunk_section()` chunks.

**Decisions:** see `docs/DECISIONS.md`. Rebuild data: `python -m mine_copilot.ingest.build`. `chunk_section()` (1,500 chars, paragraph split) is ready for M3 indexing. `SEVERITY` column is the severity filter (fatal / permanent_disability / lost_time / restricted_duty / no_lost_time / other).
