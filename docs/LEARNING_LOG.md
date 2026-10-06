# Learning Log

## Milestone 0 — Setup
- A **virtual environment** (`.venv`) isolates this project's packages from system Python.
- **`.env` + `.gitignore`**: secrets live in a local file that git never tracks; `.env.example` documents the keys.
- **`pyproject.toml`** is the single modern config for package metadata, deps, ruff and pytest.
- **`src/` layout** forces imports to go through the installed package, catching packaging bugs early.
- **CLAUDE.md** is the hand-off file Claude Code reads at the start of every session.

## Milestone 1 — Data
- **Profile before filtering:** check real `SUBUNIT` / `DEGREE_INJURY` values; they're abbreviated and messy.
- **Stratified-ish sampling:** keep all rare, high-value rows (fatalities), then a seeded random fill → reproducible.
- **Pin external sources:** eCFR fetched at a fixed date so answers and evals are reproducible.
- **Real data surprises:** eCFR needs gzip; Part 56 has duplicate-looking sections (56.5001 vs 56.5001T) — caught by a UNIQUE index.
- **Chunking by structure** (section → paragraphs) keeps the citation ID on every chunk.
- **Fixtures with negative cases** make tests meaningful, not just "it runs".
