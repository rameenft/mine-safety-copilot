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
- Update `## Status` below at the end of every milestone; `docs/LEARNING_LOG.md` is bullets only.

## Hard product rules
- Every claim cites `30 CFR §56.xxxx` or an accident `DOCUMENT_NO`.
- Not in sources → say so. Never answer from general knowledge.
- Refuse out-of-scope; resist prompt injection.
- Statistics come from DB queries and state their filters.
- Regulation answers end with a one-line not-legal-advice note.

## Stack
Python 3.13 venv at `.venv` · SQLite · Chroma + local sentence-transformers · BM25 + RRF ·
Gemini (free tier, `google-genai`) behind `llm.py` provider interface · MCP Python SDK (stdio) · pytest + ruff.

## Commands
- Activate: `source .venv/bin/activate`
- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q` · Lint: `ruff check .`

## Status
**Done:** Milestone 0 — venv, skeleton, `.gitignore`, `.env.example`, `pyproject.toml`, CLAUDE.md, docs stubs, git init.
**Next:** Milestone 1 — Data (download Accidents/Mines zips + eCFR Part 56, filter, load SQLite, fixtures, DATA_CARD).
**Decisions:** see `docs/DECISIONS.md`. GitHub: `rameenft/mine-safety-copilot` (public).
